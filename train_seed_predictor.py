import os
import time
import copy

from skimage import io, transform
import numpy as np
from matplotlib import pyplot as plt

from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch.optim import lr_scheduler
import torchvision.transforms.functional as F

cudnn.benchmark = True

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = np.inf
    best_acc = 0.0
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        for phase in ['train']: # Just train currently
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            running_loss = 0.0
            for index, batch in enumerate(dataloader):
                inputs = batch['image'].float().to(device)
                latents = batch['latents'].float().to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, latents)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            if phase == 'train':
                scheduler.step()

            if running_loss < min_loss:
                min_loss = running_loss
                print (min_loss)
                best_model_wts = copy.deepcopy(model.state_dict())
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    model.load_state_dict(best_model_wts)
    torch.save(model, 'prediction.pt')
    return model

class SeedPredictionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        assert (os.path.isdir(root_dir))
        contents = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
        self._images = [image for image in contents if image.endswith('.png')]
        self._seeds = [seed for seed in contents if seed.endswith('.txt')]
        assert (len(self._images) == len(self._seeds)), f'{len(self._images)} vs. {len(self._seeds)}'
        self.transform = transform

    def __len__(self):
        return len(self._images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = Image.open(self._images[idx])
        latents = np.loadtxt(self._seeds[idx])
        sample = {'image': image, 'latents': latents}
        if self.transform:
            sample = self.transform(sample)
        return sample

class SquarePad:
    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']
        w, h = image.size
        max_wh = np.max([w, h])
        hp = int((max_wh - w) / 2)
        vp = int((max_wh - h) / 2)
        padding = (hp, vp, hp, vp)
        return {'image': F.pad(image, padding, (255, 255, 255), 'constant'), 'latents': latents}

class Resize(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.transform = transforms.Resize(output_size)

    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']
        image = self.transform(image)
        return {'image': image, 'latents': latents}

class ToTensor(object):
    def __init__(self):
        self.transform = transforms.ToTensor()

    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']
        return {'image': self.transform(image), 'latents': torch.from_numpy(latents)}

class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        return {'image': image, 'latents': latents}

class Normalize(object):
    def __init__(self, mean, std):
        self.transform = transforms.Normalize(mean=mean, std=std)

    def __call__(self, sample):
        image, latents = sample['image'], sample['latents']
        return {'image': self.transform(image), 'latents': latents}

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

if __name__ == "__main__":
    transforms = transforms.Compose([
                    SquarePad(),
                    Resize(224),
                    ToTensor(),
                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                 ])

    dataset = SeedPredictionDataset(root_dir='train/train/', transform=transforms)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)
    assert (torch.cuda.is_available())
    device = torch.device('cuda')

    for index, batch in enumerate(dataloader):
        images = batch['image'].float()
        latents = batch['latents'].float()
        output = torchvision.utils.make_grid(images)
        imshow(output)
        plt.show()
        break

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    NETWORK_FEATURE_CARDINALITY = model.fc.in_features
    LATENT_CARDINALITY = 512
    model.fc = nn.Linear(NETWORK_FEATURE_CARDINALITY, LATENT_CARDINALITY)
    model.to('cuda')

    criterion = nn.MSELoss()

    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    model_conv = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=100)
    print ('Done')
