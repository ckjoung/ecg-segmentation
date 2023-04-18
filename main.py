import argparse
from train import train_model

def get_args():
    parser = argparse.ArgumentParser(description='Train ECG segmentation model.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-c', '--n_channels', type=int, default=32, help='Number of channels in first encoder feature map')
    parser.add_argument('-e', '--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-b', '--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight parameter for segmentation loss')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight parameter for classification loss')
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='Focal loss parameter gamma')
    parser.add_argument('--data_dir', type=str, default='./data/lobachevsky-university-electrocardiography-database-1.0.1/data', help='Path to data')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    train_model(
        args.n_channels,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.alpha,
        args.beta,
        args.focal_gamma,
        args.data_dir,
    )
