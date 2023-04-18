# %%
import os
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from datareader import load_ludb_tensors
from loss import FocalLoss
from transform import *
from model import ECGUNet3pCGM


def train(model, device, train_loader, loss_function_seg, loss_function_cls, optimizer, alpha=1.0, beta=1.0):
    model.train()
    train_loss = 0
    train_loss_seg = 0
    train_loss_cls = 0
    n_train = len(train_loader.dataset)  # total number of train data

    for data, seg_target, cls_target in tqdm(train_loader):
        # pass through model
        data, seg_target, cls_target = data.to(device), seg_target.to(device), cls_target.to(device)
        seg_output, cls_output = model(data)

        # loss calculation and backprop
        optimizer.zero_grad()
        loss_seg = loss_function_seg(seg_output, torch.argmax(seg_target, dim=1))
        loss_cls = loss_function_cls(cls_output, cls_target)
        loss = alpha * loss_seg + beta * loss_cls
        loss.backward()
        optimizer.step()

        # update cumulative loss
        train_loss += loss.item() * len(data)
        train_loss_seg += loss_seg.item() * len(data)
        train_loss_cls += loss_cls.item() * len(data)

    # divide cumulative loss by number of train data
    train_loss = train_loss / n_train
    train_loss_seg = train_loss_seg / n_train
    train_loss_cls = train_loss_cls / n_train
    return train_loss, train_loss_seg, train_loss_cls


def test(model, device, test_loader, loss_function_seg, loss_function_cls):
    model.eval()
    test_loss = 0
    test_loss_seg = 0
    test_loss_cls = 0
    n_test = len(test_loader.dataset)  # total number of test data

    for data, seg_target, cls_target in test_loader:
        data, seg_target, cls_target = data.to(device), seg_target.to(device), cls_target.to(device)
        seg_output, cls_output = model(data)

        loss_seg = loss_function_seg(seg_output, torch.argmax(seg_target, dim=1))
        loss_cls = loss_function_cls(cls_output, cls_target)
        loss = loss_seg + loss_cls

        test_loss += loss.item() * len(data)
        test_loss_seg += loss_seg.item() * len(data)
        test_loss_cls += loss_cls.item() * len(data)

    test_loss = test_loss / n_test
    test_loss_seg = test_loss_seg / n_test
    test_loss_cls = test_loss_cls / n_test
    return test_loss, test_loss_seg, test_loss_cls


def train_model(
    n_channels, 
    epochs, 
    batch_size, 
    learning_rate, 
    alpha, 
    beta, 
    focal_gamma, 
    data_dir,
    sampler=False, 
):
    
    # train/test split for ludb
    n_ludb_train = 100
    ludb_files = [os.path.abspath(os.path.join(data_dir, p))[:-4] for p in os.listdir(data_dir) if p.endswith('.hea')]
    ludb_files_train = ludb_files[:n_ludb_train]
    ludb_files_test = ludb_files[n_ludb_train:]

    X_train, y_seg_train, y_cls_train = load_ludb_tensors(ludb_files_train)
    X_test, y_seg_test, y_cls_test = load_ludb_tensors(ludb_files_test)

    # prepare training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ECGUNet3pCGM(n_channels=n_channels).to(device)
    loss_function_seg = FocalLoss(gamma=focal_gamma)
    loss_function_cls = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, verbose=True, eta_min=1e-5)

    # define sampler
    if sampler:
        target = y_cls_train
        weight = torch.tensor([1. / torch.sum(target == t) for t in torch.unique(target)])
        samples_weight = torch.tensor([weight[int(t)] for t in target]).double()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        shuffle = None
    else:
        sampler = None
        shuffle = True

    # create dataloader
    train_dataset = CustomTensorDataset(tensors=(X_train, y_seg_train, y_cls_train), transform=Compose([
        RandomCrop(2000, start=1000, end=4000),
        BaselineWander(prob=0.2),
        GaussianNoise(prob=0.2),
        PowerlineNoise(prob=0.2),
        ChannelResize(),
        BaselineShift(prob=0.2),
    ]))
    test_dataset = CustomTensorDataset(tensors=(X_test, y_seg_test, y_cls_test))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # train
    for epoch in range(1, epochs + 1):
        train_loss, train_loss_seg, train_loss_cls = train(model, device, train_loader, loss_function_seg, 
                                                           loss_function_cls, optimizer, alpha=alpha,
                                                           beta=beta)
        test_loss, test_loss_seg, test_loss_cls = test(model, device, test_loader, loss_function_seg,
                                                       loss_function_cls)

        print(f'''epoch: {epoch}/{epochs}:
            \ttrain_loss: {train_loss:.4f}, train_loss_seg: {train_loss_seg:.4f}, train_loss_cls: {train_loss_cls:.4f}, 
            \ttest_loss: {test_loss:.4f}, test_loss_seg : {test_loss_seg:.4f}, test_loss_cls : {test_loss_cls:.4f},
        ''')

        if scheduler is not None:
            scheduler.step()