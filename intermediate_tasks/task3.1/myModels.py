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

inv_transform = transforms.Compose([
                        transforms.ToPILImage(),
                                ])

class ContextEncoder(nn.Module):
    def __init__(self, lr = 3e-4, latent_space = 128):
        super(ContextEncoder, self).__init__()
        self.decoder = Decoder(1,latent_space).to(device)
        self.encoder = Encoder(1,latent_space).to(device)
        self.discriminator = Discriminator(latent_space).to(device)
        self.latent_space = latent_space

        self.criterionL1 = nn.L1Loss().to(device)
        self.criterionBCE = nn.BCELoss().to(device)

        self.optimizer_G = torch.optim.Adam(list(self.decoder.parameters()) + list(self.encoder.parameters()), lr=lr, betas=(.5, 0.999))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(.5, 0.999))
        
    def switch_off_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def switch_on_discriminator(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def train_epoch(self, e, batch_num, skeleton, target):
        # segmentation nx256x256
        fake_space = self.encoder(skeleton.to(device))
        true_space = self.encoder(target.to(device))
        fake_prediction = self.decoder(fake_space)

        if batch_num == 0 and (e%10 == 0 or e < 10):
            for i in range(1):
                s = skeleton[i].squeeze().detach().cpu().numpy()
                t = target[i].squeeze().detach().cpu().numpy()
                f = fake_prediction[i].squeeze().detach().cpu().numpy()
                fig = plt.figure(figsize = (9,3))
                plt.subplot(1,3,1)
                plt.imshow(s, cmap = 'gray')
                plt.title('Cropped Image')
                plt.axis('off')
                plt.subplot(1,3,2)
                plt.imshow(t, cmap = 'gray')
                plt.title('Target Image')
                plt.axis('off')
                plt.subplot(1,3,3)
                plt.imshow(f, cmap = 'gray')
                plt.title('Generated Image')
                plt.axis('off')
                plt.savefig('images/train/epoch{}_{}.png'.format(e, i))


        lD = self.learn_D(true_space, fake_space)
        lG = self.learn_G(fake_space, target, fake_prediction)
        return lG, lD

    def learn_D(self, true_space, fake_space):
        self.switch_on_discriminator()
        self.optimizer_D.zero_grad()
        pred_fake = self.discriminator(fake_space.detach())
        loss_D_fake = self.criterionBCE(pred_fake, torch.zeros(fake_space.shape[0],1).to(device))

        pred_real = self.discriminator(true_space.detach())
        loss_D_real = self.criterionBCE(pred_real, torch.ones(true_space.shape[0],1).to(device))

        loss_D = (loss_D_real + loss_D_fake)/2
        loss_D.backward()
        self.optimizer_D.step()
        return loss_D.item()

    def learn_G(self, fake_space, target, fake_prediction):
        self.switch_off_discriminator()
        self.optimizer_G.zero_grad()
        pred_fake = self.discriminator(fake_space)
        loss_G_latent = self.criterionBCE(pred_fake, torch.ones(fake_space.shape[0],1).to(device))
        loss_G_MSE = self.criterionL1(fake_prediction, target.to(device))*100
        loss_G = loss_G_latent + loss_G_MSE
        loss_G.backward()
        self.optimizer_G.step()
        return loss_G.item()

    def generate(self, skeleton):
        fake_space = self.encoder(skeleton.to(device))
        return self.decoder(fake_space)


class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, latent_space = 128):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
                    nn.Linear(latent_space,latent_space),
                    nn.ReLU(),
                    nn.Linear(latent_space,1),
                    nn.Sigmoid())

    def forward(self, x):
        return self.model(x)

class Encoder(nn.Module):
    def __init__(self, n_channels, latent_space):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channels,64,3,stride = 2,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,64,3,padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64,128,3,stride = 2,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128,256,3,stride = 2,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256,512,3,stride = 2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,stride=2,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,3,padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512,512,2),
          )

        self.linear = nn.Sequential(
            nn.Linear(512, latent_space),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

    def forward(self, x):
        a = self.model(x).reshape(-1,512)
        return self.linear(a)

class Decoder(nn.Module):
    def __init__(self, n_channels, latent_space):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,512,3,padding = 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Conv2d(256,256,3,padding = 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128,128,3,padding = 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,64,3,padding = 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64,n_channels,3,padding = 1),
            nn.Tanh(),
            )

        self.decoder_linear = nn.Sequential(
                            nn.Linear(latent_space, 512),
                            nn.LeakyReLU(),
                            nn.Dropout(0.2)
                        )

    def forward(self, x):
        return self.model(self.decoder_linear(x).view(-1,512,1,1))