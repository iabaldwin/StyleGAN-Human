import os
import PIL
from PIL import Image
import numpy as np
import legacy
import dnnlib
import torch
from torch.optim import Adam
import torchvision.transforms as T
from loss_model import LossNetwork

global counter
counter = 0

def show(G, w, save=False):
    # label = torch.zeros([1, G.c_dim]).to(device)
    # w = G.mapping(z, label, truncation_psi=1.0)
    W = w.repeat((16,1))
    W = torch.unsqueeze(W, 0)
    img = G.synthesis(W, noise_mode='const', force_fp32=True)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    if save:
        global counter
        counter += 1
        if not os.path.isdir('imagery'):
            os.makedirs('imagery')
        img.save('imagery/{:05d}.png'.format(counter))
    else:
        img.show()

PKL = 'pretrained_models/stylegan_human_v3_512.pkl'
save = True
device = torch.device('cuda')
with dnnlib.util.open_url(PKL) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device)
    SEED = 10
    z = torch.from_numpy(np.random.RandomState(SEED).randn(1, G.z_dim)).to(device)
    label = torch.zeros([1, G.c_dim]).to(device)
    w = G.mapping(z, label,truncation_psi=1.0)
    target_image = G.synthesis(w, force_fp32=True)
    target_image = (target_image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    target_image = PIL.Image.fromarray(target_image[0].cpu().numpy(), 'RGB')
    target_image.show()

    # Compute sample statistics
    label = torch.zeros([1, G.c_dim], device=device)
    samples = np.zeros((100, 3, 512, 256))
    for i in range(0, 100):
        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
        w = G.mapping(z, label,truncation_psi=1.0)
        img = G.synthesis(w, force_fp32=True)
        samples[i, :, : : ] = img.cpu()
    samples = torch.from_numpy(samples)
    batch_mean = torch.mean(samples, dim=(0, 2, 3))
    batch_std = torch.std(samples, dim=(0, 2, 3))
    L = LossNetwork(G, batch_mean, batch_std).to(device)


    min_loss = np.inf
    min_w = None
    show_counter = 0
    for i in range(11, 100, 10):
        z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(device)
        w = torch.squeeze(G.mapping(z, label,truncation_psi=1.0))
        w_hat = w[0,:]
        show(G, w_hat, save=False)
        w_hat.requires_grad_(True)
        optimizer = Adam([w_hat], 0.010)
        for j in range(0, 1000):
            optimizer.zero_grad()
            loss = L(w_hat, target_image)
            loss.backward()
            optimizer.step()
            if loss < min_loss:
                min_loss = loss
                min_w = w_hat
                print (i, j, min_loss)
                if (show_counter % 10 == 0):
                    show(G, w_hat, save=False)
                show_counter += 1
    show(G, min_w)
