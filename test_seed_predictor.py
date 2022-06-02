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
    z = torch.from_numpy(np.random.RandomState(10).randn(1, G.z_dim)).to(DEVICE)
    w = G.mapping(z, label, truncation_psi=1.0)
    img = G.synthesis(w, force_fp32=True)

    show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    input_image = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB');
    input_image.show()

    train_transforms = transforms.Compose([
	SquarePad(),
	transforms.Resize(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    inputs = train_transforms(input_image)
    plot(inputs)

    inputs = torch.unsqueeze(inputs, 0).to(DEVICE)

    z = model(inputs)
    w = G.mapping(z, label, truncation_psi=1.0)
    img = G.synthesis(w, force_fp32=True)
    show = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    input_image = PIL.Image.fromarray(show[0].cpu().numpy(), 'RGB');
    input_image.show()
