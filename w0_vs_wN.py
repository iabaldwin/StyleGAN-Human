import os
import time
import copy

import PIL.Image
from skimage import io, transform
import numpy as np
from matplotlib import pyplot as plt

import dnnlib
import legacy

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
import torchvision.transforms.functional as F
from loss_model import LossNetwork
from torch.optim import Adam

PKL = 'pretrained_models/stylegan_human_v3_512.pkl'
DEVICE  = torch.device('cuda')

def plot(tensor):
    show = (tensor.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB').show();

with dnnlib.util.open_url(PKL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)
    label = torch.zeros([1, G.c_dim]).to(DEVICE)
    z = torch.from_numpy(np.random.RandomState(20).randn(1, G.z_dim)).to(DEVICE)
    w = G.mapping(z, label, truncation_psi=1.0)
    img = G.synthesis(w, force_fp32=True)
    plot(img)
   
    print (w.shape)
    w = w[0]
    print (w.shape)
    w0 = w[0,:]
    print (w0.shape)
    w = w0.repeat((16, 1))
    print (w.shape)
    w = torch.unsqueeze(w, 0)
    img = G.synthesis(w, force_fp32=True)
    plot(img)
