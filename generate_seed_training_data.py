import os
import shutil
import PIL
from PIL import Image
import numpy as np
import legacy
import dnnlib
import torch

PKL = 'pretrained_models/stylegan_human_v3_512.pkl'
DEVICE = torch.device('cuda')

def generate_training_data():
    with dnnlib.util.open_url(PKL) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(DEVICE)

    if os.path.isdir('train'):
        shutil.rmtree('train')
    os.makedirs('train/train')

    seeds = range(0, 10000, 10)

    for seed_idx, seed in enumerate(seeds):
        print (seed_idx, seed)
        label = torch.zeros([1, G.c_dim], device=DEVICE)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(DEVICE)
        w = G.mapping(z, label,truncation_psi=1)
        img = G.synthesis(w, noise_mode='const', force_fp32 = True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        result = PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        result.save(f'train/train/seed_{seed:04d}.png')
        np.savetxt(f'train/train/seed_{seed:04d}.txt', z.cpu().numpy())


        if (seed_idx % 100 == 0):
            result.show()

if __name__ == "__main__":
    generate_training_data()
