import unittest

import PIL
from PIL import Image
import numpy as np
import legacy
import dnnlib
import torch
from torch.optim import Adam
from loss_model import LossNetwork

class TestPerceptualLoss(unittest.TestCase):

    PKL = 'pretrained_models/stylegan_human_v3_512.pkl'
    DEVICE  = torch.device('cuda')

    @classmethod
    def setUpClass(cls):
        with dnnlib.util.open_url(TestPerceptualLoss.PKL) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(TestPerceptualLoss.DEVICE)
        label = torch.zeros([1, G.c_dim], device=TestPerceptualLoss.DEVICE)
        samples = np.zeros((100, 3, 512, 256))
        for i in range(0, 100):
            z = torch.from_numpy(np.random.RandomState(i).randn(1, G.z_dim)).to(TestPerceptualLoss.DEVICE)
            w = G.mapping(z, label,truncation_psi=1.0)
            img = G.synthesis(w, force_fp32=True)
            samples[i, :, : : ] = img.cpu()
        samples = torch.from_numpy(samples)
        cls.batch_mean = torch.mean(samples, dim=(0, 2, 3))
        cls.batch_std = torch.std(samples, dim=(0, 2, 3))
        cls.G = G

    def to_PIL(self, z):
        label = torch.zeros([1, self.G.c_dim]).to(TestPerceptualLoss.DEVICE)
        w = self.G.mapping(z, label,truncation_psi=1.0)
        image = self.G.synthesis(w, force_fp32=True)
        image = (image.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        image = PIL.Image.fromarray(image[0].cpu().numpy(), 'RGB')
        return image

    def test_close_optimize(self):
        SEED = 10
        z_target = torch.from_numpy(np.random.RandomState(SEED).randn(1, self.G.z_dim)).to(TestPerceptualLoss.DEVICE)
        z_hat = z_target + torch.from_numpy(0.5 * np.random.RandomState(2).randn(1, self.G.z_dim)).to(TestPerceptualLoss.DEVICE)
        z_hat.requires_grad_(True)
        L = LossNetwork(self.G, self.batch_mean, self.batch_std).to(TestPerceptualLoss.DEVICE)
        optimizer = Adam([z_hat], 0.010)

        target_image = self.to_PIL(z_target)
        target_image.show()
        self.to_PIL(z_hat).show()

        min_loss = None
        for j in range(0, 100):
            optimizer.zero_grad()
            loss = L(z_hat, target_image)
            loss.backward()
            optimizer.step()
            if not min_loss or (loss < min_loss):
                min_loss = loss
        self.assertLess(min_loss, 0.5)
        self.to_PIL(z_hat).show()

if __name__ == '__main__':
    unittest.main()
