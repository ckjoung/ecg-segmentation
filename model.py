# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


# %%
class ConvBnRelu1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU()
        self.do = nn.Dropout1d(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.do(x)
        return x


class StackEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv1 = ConvBnRelu1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x, self.pool(x)


class StackDecoder(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=8, stride=2, padding=3)
        self.conv1 = ConvBnRelu1d(in_channels + skip_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = ConvBnRelu1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x, skip):
        x = self.up(x)
        if skip.shape[2] != x.shape[2]:
            x = F.pad(x, (0, 1))  # pad last dimension of x by (0,1)
        x = torch.cat((x, skip), dim=1)  # concatenate along channel dimension
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class StackDecoder3p(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=9, padding=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels[0], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_channels[1], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(in_channels[2], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv4 = nn.Conv1d(in_channels[3], skip_channels, kernel_size=kernel_size, padding=padding)
        self.conv5 = nn.Conv1d(in_channels[4], skip_channels, kernel_size=kernel_size, padding=padding)
        self.aggregate = ConvBnRelu1d(skip_channels * 5, out_channels, kernel_size=kernel_size, padding=padding)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x3)
        x4 = self.conv4(x4)
        x5 = self.conv5(x5)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)  # concatenate along channel dimension
        x = self.aggregate(x)  # feature aggregation
        return x


class ECGUNet(nn.Module):
    def __init__(self, n_channels=4):
        super().__init__()

        filters = [n_channels * (2 ** n) for n in range(5)]  # n_filters for encoder feature maps
        self.down1 = StackEncoder(1, filters[0])
        self.down2 = StackEncoder(filters[0], filters[1])
        self.down3 = StackEncoder(filters[1], filters[2])
        self.down4 = StackEncoder(filters[2], filters[3])

        self.up4 = StackDecoder(filters[4], filters[3], filters[3])
        self.up3 = StackDecoder(filters[3], filters[2], filters[2])
        self.up2 = StackDecoder(filters[2], filters[1], filters[1])
        self.up1 = StackDecoder(filters[1], filters[0], filters[0])

        self.middle = nn.Sequential(ConvBnRelu1d(filters[3], filters[4]), ConvBnRelu1d(filters[4], filters[4]))
        self.classify = nn.Conv1d(filters[0], 4, kernel_size=1, padding=0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.middle(x)

        x = self.up4(x, skip4)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return self.classify(x)


class ECGUNet3p(nn.Module):
    def __init__(self, n_channels=4):
        super().__init__()

        filters = [n_channels * (2 ** n) for n in range(5)]  # n_filters for encoder feature maps
        filters_skip = filters[0]  # n_filters for skip connections
        filters_decoder = filters_skip * 5  # n_filters for decoder feature maps

        self.down1 = StackEncoder(1, filters[0])
        self.down2 = StackEncoder(filters[0], filters[1])
        self.down3 = StackEncoder(filters[1], filters[2])
        self.down4 = StackEncoder(filters[2], filters[3])
        self.middle = nn.Sequential(ConvBnRelu1d(filters[3], filters[4]), ConvBnRelu1d(filters[4], filters[4]))

        self.up4 = StackDecoder3p(filters, filters_skip, filters_decoder)
        self.up3 = StackDecoder3p(filters[:3] + [filters_decoder] * 1 + filters[4:], filters_skip, filters_decoder)
        self.up2 = StackDecoder3p(filters[:2] + [filters_decoder] * 2 + filters[4:], filters_skip, filters_decoder)
        self.up1 = StackDecoder3p(filters[:1] + [filters_decoder] * 3 + filters[4:], filters_skip, filters_decoder)
        self.segment = nn.Conv1d(filters_decoder, 4, kernel_size=1, padding=0)

    def forward(self, x):
        # encoder
        X_enc1, x = self.down1(x)
        X_enc2, x = self.down2(x)
        X_enc3, x = self.down3(x)
        X_enc4, x = self.down4(x)
        X_enc5 = self.middle(x)

        # decoder
        X_dec5 = X_enc5
        X_dec4 = self.up4(
            F.max_pool1d(X_enc1, kernel_size=8, stride=8),
            F.max_pool1d(X_enc2, kernel_size=4, stride=4),
            F.max_pool1d(X_enc3, kernel_size=2, stride=2),
            X_enc4,
            F.interpolate(X_dec5, size=X_enc4.shape[-1], mode='linear', align_corners=False)
        )
        X_dec3 = self.up3(
            F.max_pool1d(X_enc1, kernel_size=4, stride=4),
            F.max_pool1d(X_enc2, kernel_size=2, stride=2),
            X_enc3,
            F.interpolate(X_dec4, size=X_enc3.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc3.shape[-1], mode='linear', align_corners=False)
        )
        X_dec2 = self.up2(
            F.max_pool1d(X_enc1, kernel_size=2, stride=2),
            X_enc2,
            F.interpolate(X_dec3, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc2.shape[-1], mode='linear', align_corners=False)
        )
        X_dec1 = self.up1(
            X_enc1,
            F.interpolate(X_dec2, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec3, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc1.shape[-1], mode='linear', align_corners=False)
        )
        return self.segment(X_dec1)


class ResnetBlock(nn.Module):
    def __init__(self, n_channel, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channel, n_channel, kernel_size, 1, padding)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size, 1, padding)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.lrl = nn.LeakyReLU()
        self.do = nn.Dropout1d(0.2)

    def forward(self, x):
        y = self.do(self.lrl(self.bn1(self.conv1(x))))
        y = self.bn2(self.conv2(y))
        return x + y


class ECGUNet3pCGM(nn.Module):
    def __init__(self, n_channels=4, mask=True):
        super().__init__()
        self.mask = mask 

        filters = [n_channels * (2 ** n) for n in range(5)]  # n_filters for encoder feature maps
        filters_skip = filters[0]  # n_filters for skip connections
        filters_decoder = filters_skip * 5  # n_filters for decoder feature maps

        self.down1 = StackEncoder(1, filters[0])
        self.down2 = StackEncoder(filters[0], filters[1])
        self.down3 = StackEncoder(filters[1], filters[2])
        self.down4 = StackEncoder(filters[2], filters[3])
        self.middle = nn.Sequential(ConvBnRelu1d(filters[3], filters[4]), ConvBnRelu1d(filters[4], filters[4]))

        self.classify = nn.Sequential(
            nn.BatchNorm1d(sum(filters)),
            nn.LeakyReLU(),
            nn.Conv1d(sum(filters), 512, 17, 1, 8),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout1d(0.2),
            nn.Conv1d(512, 512, 17, 1, 8),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1),
            nn.Linear(512, 2)
        )

        self.up4 = StackDecoder3p(filters, filters_skip, filters_decoder)
        self.up3 = StackDecoder3p(filters[:3] + [filters_decoder] * 1 + filters[4:], filters_skip, filters_decoder)
        self.up2 = StackDecoder3p(filters[:2] + [filters_decoder] * 2 + filters[4:], filters_skip, filters_decoder)
        self.up1 = StackDecoder3p(filters[:1] + [filters_decoder] * 3 + filters[4:], filters_skip, filters_decoder)
        self.segment = nn.Conv1d(filters_decoder, 4, kernel_size=1, padding=0)

    def apply_cls_mask(self, seg, cls):
        cls_mask = (cls == 0).float()  # 0 if label is not 0
        seg_masked = torch.stack((
            torch.einsum('bt,b->bt', seg[:, 0, :], cls_mask),  # P
            seg[:, 1, :],  # QRS
            seg[:, 2, :],  # T
            seg[:, 3, :],  # None
        ), dim=1)  # (B,4,len_wave)

        return seg_masked

    def forward(self, x):
        # encoder
        X_enc1, x = self.down1(x)
        X_enc2, x = self.down2(x)
        X_enc3, x = self.down3(x)
        X_enc4, x = self.down4(x)
        X_enc5 = self.middle(x)

        # classification
        aggregate = torch.cat([
            F.avg_pool1d(X_enc1, 16),
            F.avg_pool1d(X_enc2, 8),
            F.avg_pool1d(X_enc3, 4),
            F.avg_pool1d(X_enc4, 2),
            X_enc5
        ], dim=1)
        X_cls_prob = self.classify(aggregate)
        X_cls = X_cls_prob.argmax(dim=1)  # (B,)

        # decoder
        X_dec5 = X_enc5
        X_dec4 = self.up4(
            F.max_pool1d(X_enc1, kernel_size=8, stride=8),
            F.max_pool1d(X_enc2, kernel_size=4, stride=4),
            F.max_pool1d(X_enc3, kernel_size=2, stride=2),
            X_enc4,
            F.interpolate(X_dec5, size=X_enc4.shape[-1], mode='linear', align_corners=False)
        )
        X_dec3 = self.up3(
            F.max_pool1d(X_enc1, kernel_size=4, stride=4),
            F.max_pool1d(X_enc2, kernel_size=2, stride=2),
            X_enc3,
            F.interpolate(X_dec4, size=X_enc3.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc3.shape[-1], mode='linear', align_corners=False)
        )
        X_dec2 = self.up2(
            F.max_pool1d(X_enc1, kernel_size=2, stride=2),
            X_enc2,
            F.interpolate(X_dec3, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc2.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc2.shape[-1], mode='linear', align_corners=False)
        )
        X_dec1 = self.up1(
            X_enc1,
            F.interpolate(X_dec2, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec3, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec4, size=X_enc1.shape[-1], mode='linear', align_corners=False),
            F.interpolate(X_dec5, size=X_enc1.shape[-1], mode='linear', align_corners=False)
        )

        X_seg = self.segment(X_dec1)

        if self.mask and not self.training:
            X_seg = self.apply_cls_mask(X_seg, X_cls)
            
        return X_seg, X_cls_prob
