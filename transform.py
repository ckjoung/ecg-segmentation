#%%
import random
import math
import torch
from torch.utils.data import Dataset

def Tnoise_powerline(fs=100, N=1000,C=1,fn=50.,K=3, channels=1):
    '''powerline noise inspired by https://ieeexplore.ieee.org/document/43620
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale: 1)
    fn: base frequency of powerline noise (Hz)
    K: number of higher harmonics to be considered
    channels: number of output channels (just rescaled by a global channel-dependent factor)
    '''
    #C *= 0.333 #adjust default scale
    t = torch.arange(0,N/fs,1./fs)
    
    signal = torch.zeros(N)
    phi1 = random.uniform(0,2*math.pi)
    for k in range(1,K+1):
        ak = random.uniform(0,1)
        signal += C*ak*torch.cos(2*math.pi*k*fn*t+phi1)
    signal = C*signal[:,None]
    if(channels>1):
        channel_gains = torch.empty(channels).uniform_(-1,1)
        signal = signal*channel_gains[None]
    return signal

def Tnoise_baseline_wander(fs=100, N=1000, C=1.0, fc=0.5, fdelta=0.01,channels=1,independent_channels=False):
    '''baseline wander as in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5361052/
    fs: sampling frequency (Hz)
    N: lenght of the signal (timesteps)
    C: relative scaling factor (default scale : 1)
    fc: cutoff frequency for the baseline wander (Hz)
    fdelta: lowest resolvable frequency (defaults to fs/N if None is passed)
    channels: number of output channels
    independent_channels: different channels with genuinely different outputs (but all components in phase) instead of just a global channel-wise rescaling
    '''
    if(fdelta is None):# 0.1
        fdelta = fs/N

    K = int((fc/fdelta)+0.5)
    t = torch.arange(0, N/fs, 1./fs).repeat(K).reshape(K, N)
    k = torch.arange(K).repeat(N).reshape(N, K).T
    phase_k = torch.empty(K).uniform_(0, 2*math.pi).repeat(N).reshape(N, K).T
    a_k = torch.empty(K).uniform_(0, 1).repeat(N).reshape(N, K).T
    pre_cos = 2*math.pi * k * fdelta * t + phase_k
    cos = torch.cos(pre_cos)
    weighted_cos = a_k * cos
    res = weighted_cos.sum(dim=0)
    return C*res

class Compose():
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, wave, target):
        for t in self.transforms:
            wave, target = t(wave, target)
        return wave, target

class RandomCrop():
    def __init__(self, length, start, end):
        self.length = length
        self.start = start
        self.end = end
    
    def __call__(self, wave, target):
        start = random.randint(self.start, self.end-self.length)
        end = start + self.length
        return wave[:,start:end], target[:,start:end]

class ChannelResize():
    def __init__(self, magnitude_range=(0.5, 2)):
        self.log_magnitude_range = torch.log(torch.tensor(magnitude_range))

    def __call__(self, wave, target):
        channels, len_wave = wave.shape
        resize_factors = torch.exp(torch.empty(channels).uniform_(*self.log_magnitude_range)) 
        resize_factors = resize_factors.repeat(len_wave).view(wave.T.shape).T 
        wave = resize_factors * wave
        return wave, target
    
class GaussianNoise():
    def __init__(self, prob=1.0, scale=0.01):
        self.scale = scale
        self.prob = prob
    
    def __call__(self, wave, target):
        if random.random() < self.prob:
            wave += self.scale * torch.randn(wave.shape)
        return wave, target
    
class BaselineShift():
    def __init__(self, prob=1.0, scale=1.0):
        self.prob = prob
        self.scale = scale

    def __call__(self, wave, target):
        if random.random() < self.prob:
            shift = torch.randn(1)
            wave = wave + self.scale * shift
        return wave, target
    
class BaselineWander():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            wander = Tnoise_baseline_wander(fs=self.freq, N=len_wave) 
            wander = wander.repeat(channels).view(wave.shape)
            wave = wave + wander
        return wave, target

class PowerlineNoise():
    def __init__(self, prob=1.0, freq=500):
        self.freq = freq
        self.prob = prob

    def __call__(self, wave, target):
        if random.random() < self.prob:
            channels, len_wave = wave.shape
            noise = Tnoise_powerline(fs=self.freq, N=len_wave, channels=channels).T 
            wave = wave + noise
        return wave, target

class CustomTensorDataset(Dataset):
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]

        if self.transform:
            x, y = self.transform(x, y)

        return (x, y) + tuple(tensor[index] for tensor in self.tensors[2:])

    def __len__(self):
        return self.tensors[0].size(0)