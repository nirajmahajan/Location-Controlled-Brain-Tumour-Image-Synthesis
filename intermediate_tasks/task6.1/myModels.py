import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models
from torchvision.models import vgg16, vgg16_bn, alexnet
from torch import nn, optim
from torch.nn import functional as F
import pickle
import matplotlib.pyplot as plt
import argparse
import sys
import os
import functools
import PIL

# reference https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Identity(nn.Module):
    def forward(self, x):
        return x

class DCGAN(nn.Module):
    def __init__(self, lr = 2e-4):
        super(DCGAN, self).__init__()
        self.generator = Generator((1,256,256)).to(device)
        self.discriminator = Discriminator((1,256,256)).to(device)

        self.criterionBCE = nn.BCELoss().to(device)

        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        
    def switch_off_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def generate(self, skeleton):
        return self.generator(skeleton.to(device))

    def train_epoch(self, e, batch_num, target):
        noise = torch.randn(target.shape[0],32).to(device)
        fake_images = self.generator(noise)

        if batch_num == 0 and (e%10 == 0 or e < 10):
            for i in range(1):
                t = target[i].squeeze().detach().cpu().numpy()
                f = fake_images[i].squeeze().detach().cpu().numpy()
                fig = plt.figure(figsize = (6,3))
                plt.subplot(1,2,1)
                plt.imshow(t, cmap = 'gray')
                plt.title('Dataset Image')
                plt.axis('off')
                plt.subplot(1,2,2)
                plt.imshow(f, cmap = 'gray')
                plt.title('Generated Tumor')
                plt.axis('off')
                plt.savefig('images/train/epoch{}_{}.png'.format(e, i))


        lD = self.learn_D(fake_images, target.to(device))
        lG = self.learn_G(fake_images)
        return lG, lD

    def generate(self, n):
        noise = torch.randn(n,32).to(device)
        return self.generator(noise)

    def learn_D(self, fake_images, target):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        pred_fake = self.discriminator(fake_images.detach())
        loss_D_fake = self.criterionBCE(pred_fake, torch.zeros(fake_images.shape[0],1).to(device))

        pred_real = self.discriminator(target.detach())
        loss_D_real = self.criterionBCE(pred_real, torch.ones(fake_images.shape[0],1).to(device))

        loss_D = (loss_D_real + loss_D_fake)/2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.item()

    def learn_G(self, fake_images):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        pred_fake = self.discriminator(fake_images)
        loss_G = self.criterionBCE(pred_fake, torch.ones(fake_images.shape[0],1).to(device))
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()

# class UNetDown(nn.Module):
#     def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
#         super(UNetDown, self).__init__()
#         model = [nn.Conv2d(in_size, out_size, 4, stride=2, padding=1, bias=False)]
#         if normalize:
#             model.append(nn.BatchNorm2d(out_size, 0.8))
#         model.append(nn.LeakyReLU(0.2))
#         if dropout:
#             model.append(nn.Dropout(dropout))

#         self.model = nn.Sequential(*model)

#     def forward(self, x):
#         return self.model(x)


# class UNetUp(nn.Module):
#     def __init__(self, in_size, out_size, dropout=0.0):
#         super(UNetUp, self).__init__()
#         model = [
#             nn.ConvTranspose2d(in_size, out_size, 4, stride=2, padding=1, bias=False),
#             nn.BatchNorm2d(out_size, 0.8),
#             nn.ReLU(inplace=True),
#         ]
#         if dropout:
#             model.append(nn.Dropout(dropout))

#         self.model = nn.Sequential(*model)

#     def forward(self, x, skip_input):
#         x = self.model(x)
#         out = torch.cat((x, skip_input), 1)
#         return out


# class Generator(nn.Module):
#     def __init__(self, input_shape = (1,64,64)):
#         super(Generator, self).__init__()
#         channels, _, _ = input_shape
#         self.down1 = UNetDown(channels, 64, normalize=False)
#         self.down2 = UNetDown(64, 128)
#         self.down3 = UNetDown(128, 256, dropout=0.5)
#         self.down4 = UNetDown(256, 512, dropout=0.5)
#         self.down5 = UNetDown(512, 512, dropout=0.5)

#         self.up2 = UNetUp(512, 512, dropout=0.5)
#         self.up3 = UNetUp(1024, 256, dropout=0.5)
#         self.up4 = UNetUp(512, 128)
#         self.up5 = UNetUp(256, 64)

#         final = [nn.Upsample(scale_factor=2), nn.Conv2d(128, channels, 3, 1, 1), nn.Sigmoid()]
#         self.final = nn.Sequential(*final)
#         self.inp = nn.Sequential(nn.Linear(128,1024), nn.ReLU())

#     def forward(self, x):
#         # U-Net generator with skip connections from encoder to decoder
#         x = self.inp(x).reshape(-1,1,64,64)
#         d1 = self.down1(x)
#         d2 = self.down2(d1)
#         d3 = self.down3(d2)
#         d4 = self.down4(d3)
#         d5 = self.down5(d4)
#         u2 = self.up2(d5, d4)
#         u3 = self.up3(u2, d3)
#         u4 = self.up4(u3, d2)
#         u5 = self.up5(u4, d1)

#         return self.final(u5)

class Generator(nn.Module):
    def __init__(self, input_shape = (1,256,256)):
        super(Generator, self).__init__()
        channels, _, _ = input_shape
        self.model = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(64, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(32, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(8, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(4, 0.8),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(4, 1, 4, stride=2, padding=1, bias=False),
                nn.Tanh(),
            )

        self.inp = nn.Sequential(nn.Linear(32,512), nn.ReLU())

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        x = self.inp(x).reshape(-1,512,1,1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_shape = (1,256,256)):
        super(Discriminator, self).__init__()

        channels, height, width = input_shape
        # Calculate output of image discriminator (PatchGAN)
        patch_h, patch_w = int(height / 2 ** 3), int(width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, stride, normalize):
            """Returns layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = channels
        for out_filters, stride, normalize in [(16,2,False),(32,2,True),(64, 2, True), (128, 2, True), (256, 2, True), (512, 1, True)]:
            layers.extend(discriminator_block(in_filters, out_filters, stride, normalize))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, out_filters//32, 3, 1, 1))

        self.model = nn.Sequential(*layers)

        self.linear = nn.Sequential(nn.Linear(1024,1), nn.Sigmoid())

    def forward(self, img):
        return self.linear(self.model(img).reshape(-1,1024))