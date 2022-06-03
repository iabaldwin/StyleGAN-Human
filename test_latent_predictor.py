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

model = torch.load('prediction.pt')
model.eval()

PKL = 'pretrained_models/stylegan_human_v3_512.pkl'
DEVICE  = torch.device('cuda')

class SquarePad:
    def __call__(self, image):
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, (255, 255, 255), 'constant')

def plot(tensor):
    show = (tensor.permute(1, 2, 0) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    PIL.Image.fromarray(show.cpu().numpy(), 'RGB').show();

with dnnlib.util.open_url(PKL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)
    label = torch.zeros([1, G.c_dim]).to(DEVICE)
   
    for i in range(0,10):
        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(DEVICE)
        w = G.mapping(z, label, truncation_psi=1.0)
        img = G.synthesis(w, force_fp32=True)
        show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        input_image = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB');
        input_image.show()

        image_transforms = transforms.Compose([
            SquarePad(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        inputs = image_transforms(input_image)
        # plot(inputs)
        inputs = torch.unsqueeze(inputs, 0).to(DEVICE)

        w = model(inputs)
        W = w.repeat((16, 1))
        W = torch.unsqueeze(W, 0)
        img = G.synthesis(W, force_fp32=True)
        show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        result = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB');
        result.show()

    import sys
    sys.exit()

    target_image = PIL.Image.open('young-woman-fashion-portrait-on-white-background-anna-bryukhanova.jpg')
    target_image.show() 
    inputs = image_transforms(target_image)
    plot(inputs)
    inputs = torch.unsqueeze(inputs, 0).to(DEVICE)
    
    w = model(inputs)
    # w = G.mapping(z, label, truncation_psi=1.0)
    # W(1) 
    w = np.tile(w.cpu().detach().numpy(), (16, 1))
    w = torch.from_numpy(w)
    w = torch.unsqueeze(w, 0).to(DEVICE)
    img = G.synthesis(w, force_fp32=True)
    show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    input_image = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB')
    input_image.show()

    samples = np.zeros((100, 3, 512, 256))
    for i in range(0, 100):
        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(DEVICE)
        w = G.mapping(z, label,truncation_psi=1.0)
        img = G.synthesis(w, force_fp32=True)
        samples[i, :, : : ] = img.cpu()
    samples = torch.from_numpy(samples)
    batch_mean = torch.mean(samples, dim=(0, 2, 3))
    batch_std = torch.std(samples, dim=(0, 2, 3))

    # z_hat = z
    # z_hat.requires_grad_(True)
    w_hat = w[:,0]
    w_hat.requires_grad_(True)
    L = LossNetwork(G, batch_mean, batch_std).to(DEVICE)
    optimizer = Adam([w_hat], 0.010)

    min_loss = np.inf
    for j in range(0, 100):
        optimizer.zero_grad()
        loss = L(w_hat, input_image)
        loss.backward()
        optimizer.step()
        if (loss < min_loss):
            print (min_loss)
            min_loss = loss

    # w = G.mapping(z_hat, label, truncation_psi=1.0)
    W = w_hat.repeat((16, 1))
    W = torch.unsqueeze(W, 0)
    img = G.synthesis(W, force_fp32=True)
    show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    input_image = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB')
    input_image.show()
