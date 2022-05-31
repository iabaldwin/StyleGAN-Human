import math
import torch
import torchvision.models.vgg as vgg
from torch.nn import functional as F
import torchvision.transforms as T

class LossNetwork(torch.nn.Module):
    def __init__(self, G, batch_mean, batch_std, apply_pixel_loss=False):
        super(LossNetwork, self).__init__()
        layer = 8
        self.model = vgg.vgg19(pretrained=True).features[:layer+1]
        self.model.eval()
        self.model.requires_grad_(False)
        self.G = G
        self.device = torch.device('cuda')
        self.batch_mean = batch_mean
        self.batch_std = batch_std
        self._apply_pixel_loss = apply_pixel_loss

    def get_features(self, input):
        return self.model(input)

    def forward(self, input, target_image):
        label = torch.zeros([1, self.G.c_dim]).to(self.device)
        # Compute latent code
        latent = self.G.mapping(input, label, 1.0)
        # Synthesize image
        synthesized = self.G.synthesis(latent, force_fp32 = True)

        target = T.ToTensor()(target_image).cuda()
        target = T.Normalize(mean=self.batch_mean, std=self.batch_std)(target)
        target = torch.unsqueeze(target, 0).to(self.device)

        perceptual_loss = F.mse_loss(self.get_features(synthesized), self.get_features(target), reduction='mean')
        if (self._apply_pixel_loss):
            perceptual_loss += (1.0/math.sqrt(synthesized.shape.numel())) * F.mse_loss(synthesized, target)
        return perceptual_loss
