# Deep learning based ECG segmentation for delineation of diverse arrhythmias

This repository contains the PyTorch implementation of the segmentation model and the training scripts for the paper:

> Joung C, Kim M, Paik T, Kong S-H, Oh S-Y, Jeon WK, Jeon J-h, Hong J-S, Kim W-J, Kook W, Cha M-J, van Koert O. (2024) **Deep learning based ECG segmentation for delineation of diverse arrhythmias.** *PLOS ONE* 19(6): e0303178. https://doi.org/10.1371/journal.pone.0303178

## Abstract

Accurate delineation of key waveforms in an ECG is a critical step in extracting relevant features to support the diagnosis and treatment of heart conditions. Although deep learning based methods using segmentation models to locate P, QRS, and T waves have shown promising results, their ability to handle arrhythmias has not been studied in any detail. In this paper we investigate the effect of arrhythmias on delineation quality and develop strategies to improve performance in such cases. We introduce a U-Net-like segmentation model for ECG delineation with a particular focus on diverse arrhythmias. This is followed by a post-processing algorithm which removes noise and automatically determines the boundaries of P, QRS, and T waves. Our model has been trained on a diverse dataset and evaluated against the LUDB and QTDB datasets to show strong performance, with F1-scores exceeding 99% for QRS and T waves, and over 97% for P waves in the LUDB dataset. Furthermore, we assess various models across a wide array of arrhythmias and observe that models with a strong performance on standard benchmarks may still perform poorly on arrhythmias that are underrepresented in these benchmarks, such as tachycardias. We propose solutions to address this discrepancy.

## Repository structure

```
ecg-segmentation/
├── main.py            
├── train.py           # train and test loops
├── model.py           # segmentation models
├── datareader.py      # reads LUDB records and annotations into PyTorch tensors
├── loss.py            # focal loss used as the segmentation loss
├── transform.py       # data augmentation transforms 
├── requirements.txt   
└── LICENSE
```

**`model.py`** contains three models:

- `ECGUNet`: a plain U-Net adapted to 1D signals.
- `ECGUNet3p`: a UNet 3+ model with full-scale skip connections between encoder and decoder.
- `ECGUNet3pCGM`: the `ECGUNet3p` plus an arrhythmia classification branch 

All models take a single-lead ECG of shape `(batch, 1, length)` and output per-sample scores for four classes: P wave, QRS complex, T wave, and none.

## Usage

Create the environment from the conda specification:

```bash
conda create --name ecg-seg --file requirements.txt
conda activate ecg-seg
```

Download the [Lobachevsky University Electrocardiography Database (LUDB) v1.0.1](https://physionet.org/content/ludb/1.0.1/) from PhysioNet and extract it into `./data`, so that the records are located at `./data/lobachevsky-university-electrocardiography-database-1.0.1/data`. Then run:

```bash
python main.py --data_dir ./data/lobachevsky-university-electrocardiography-database-1.0.1/data
```

| Argument | Default | Description |
| --- | --- | --- |
| `-c`, `--n_channels` | 32 | number of channels in the first encoder feature map |
| `-e`, `--epochs` | 500 | number of epochs |
| `-b`, `--batch-size` | 16 | batch size |
| `-l`, `--learning-rate` | 1e-3 | learning rate |
| `--alpha` | 1.0 | weight of the segmentation loss |
| `--beta` | 1.0 | weight of the classification loss |
| `--focal_gamma` | 1.0 | focal loss parameter γ |
| `--data_dir` | `./data/lobachevsky-university-electrocardiography-database-1.0.1/data` | path to the LUDB records |

## Acknowledgements

The powerline noise and baseline wander functions in `transform.py` are from [ecg-selfsupervised](https://github.com/tmehari/ecg-selfsupervised) repository which accompanies reference [44] of our paper:

> [44] Mehari T, Strodthoff N. Self-supervised representation learning from 12-lead ECG data. *Computers in Biology and Medicine*. 2022;141:105114. https://doi.org/10.1016/j.compbiomed.2021.105114

## Citation

```bibtex
@article{joung2024deep,
  title     = {Deep learning based ECG segmentation for delineation of diverse arrhythmias},
  author    = {Joung, Chankyu and Kim, Mijin and Paik, Taejin and Kong, Seong-Ho and Oh, Seung-Young and Jeon, Won Kyeong and Jeon, Jae-hu and Hong, Joong-Sik and Kim, Wan-Joong and Kook, Woong and Cha, Myung-Jin and van Koert, Otto},
  journal   = {PLOS ONE},
  volume    = {19},
  number    = {6},
  pages     = {e0303178},
  year      = {2024},
  publisher = {Public Library of Science},
  doi       = {10.1371/journal.pone.0303178}
}
```