import numpy as np
import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
import os
import PIL
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def check_binary(inp):
    a = (inp.clone()*0) + 0.5
    a[inp > 0.7] = 1
    a[inp < 0.3] = 0
    return ((a == 0.5).sum(1).sum(1) < 30)

class BRATS_T1c(Dataset):
    """docstring for BRATS_T1c"""
    def __init__(self, train_ratio = 0.75, transform = transforms.Resize(256), verticalflip = True, loadRectifier = True):
        super(BRATS_T1c, self).__init__()
        print('Loading data .... ', end = '', flush = True)
        self.data = torch.load('/content/drive/Shareddrives/Datasets/BRATS/data_t1c_ot.pth')['data']
        self.data = torch.swapaxes(self.data, 1,4).reshape(-1,240,240,2)
        self.corrected = torch.zeros((self.data.shape[0],1,256,256))
        self.memoised = np.zeros(self.data.shape[0]) == 1
        self.badvals = torch.arange(self.data.shape[0])[self.data.isnan().any(1).any(1).any(1)]
        print('   Done', end = '\n', flush = True)

        self.train_mode = None
        self.loadRectifier = loadRectifier
        self.healthy_mode = False
        self.get_cropped_mode = False
        self.transform = transform
        self.tumor_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize((64,64)),
                        transforms.ToTensor(),
                                        ])
        self.verticalflip = verticalflip

        if self.loadRectifier:
            from rectifier import Rectifier
            self.rectifier = Rectifier().to(device)
            model_state = torch.load('/content/drive/Shareddrives/MIC_Project_abhinav_niraj/tasks/2.1/state.pth')['state']
            dic = torch.load('/content/drive/Shareddrives/MIC_Project_abhinav_niraj/tasks/2.1/checkpoint_{}.pth'.format(model_state))
            self.rectifier.load_state_dict(dic['rectifier'])
            for name,child in self.rectifier.named_children():
                for param in child.parameters():
                    param.requires_grad = False

        else:
            self.rectifier = None


        print('Generating indices .... ', end = '', flush = True)

        healthy_indices = ((self.data[:,:,:,1]-0.5).abs() < 0.2).sum(1).sum(1) < 30
        unhealthy_indices = torch.arange(self.data.shape[0])[healthy_indices == 0]
        healthy_indices = torch.arange(self.data.shape[0])[healthy_indices == 1]
        self.healthy_train_indices = healthy_indices[0:int(healthy_indices.shape[0]*train_ratio)]
        self.healthy_test_indices = healthy_indices[int(healthy_indices.shape[0]*train_ratio):]

        self.unhealthy_train_indices = unhealthy_indices[0:int(unhealthy_indices.shape[0]*train_ratio)]
        self.unhealthy_test_indices = unhealthy_indices[int(unhealthy_indices.shape[0]*train_ratio):]

        self.combined_train_indices = torch.cat((self.healthy_train_indices, self.unhealthy_train_indices)).sort()[0]
        self.combined_test_indices = torch.cat((self.healthy_test_indices, self.unhealthy_test_indices)).sort()[0]

        self.combined_healthy_indices = torch.cat((self.healthy_train_indices, self.healthy_test_indices)).sort()[0]
        self.combined_unhealthy_indices = torch.cat((self.unhealthy_train_indices, self.unhealthy_test_indices)).sort()[0]
        self.combined_indices = torch.arange(self.data.shape[0])

        self.current_indices = np.setdiff1d(self.combined_indices, self.badvals)
        print('   Done', end = '\n', flush = True)

    def fetch_rectified(self, index, cropped):
        if self.memoised[index]:
            return self.corrected[index]
        else:
            self.memoised[index] = True
            self.rectifier.eval()
            self.corrected[index] = self.rectifier.generate(self.transform(cropped.squeeze().unsqueeze(0).unsqueeze(0)).to(device)).squeeze().unsqueeze(0)
            return self.corrected[index]

    def cropped(self,a1,a2, index, skip_if_binary = False):
        if skip_if_binary and check_binary(a2):
            return a1
        rr = torch.arange(a2.shape[1])[(a2>0.5).sum(1) > 0]
        cc = torch.arange(a2.shape[0])[(a2>0.5).sum(0) > 0]
        rmin = rr.min()
        rmax = rr.max()
        cmin = cc.min()
        cmax = cc.max()
        anew = a1.clone()
        anew[rmin:rmax,cmin:cmax] = 0

        rectified = self.fetch_rectified(index, anew)
        mask = torch.zeros(anew.shape)
        mask[rmin:rmax,cmin:cmax] = 1
        tumor = a1[rmin:rmax,cmin:cmax]
        good = rectified[0,rmin:rmax,cmin:cmax]
        return anew, mask, tumor, good, rectified

    def set_train(self, train = True):
        self.train_mode = train
        self.recompute_indices()

    def set_healthy(self, healthy = True):
        self.healthy_mode = healthy
        self.recompute_indices()

    def recompute_indices(self):
        if self.healthy_mode is None:
            if self.train_mode is None:
                self.current_indices = self.combined_indices
            elif self.train_mode:
                self.current_indices = self.combined_train_indices
            else:
                self.current_indices = self.combined_test_indices
        elif self.healthy_mode:
            if self.train_mode is None:
                self.current_indices = self.combined_healthy_indices
            elif self.train_mode:
                self.current_indices = self.healthy_train_indices
            else:
                self.current_indices = self.healthy_test_indices
        else:
            if self.train_mode is None:
                self.current_indices = self.combined_unhealthy_indices
            elif self.train_mode:
                self.current_indices = self.unhealthy_train_indices
            else:
                self.current_indices = self.unhealthy_test_indices
        self.current_indices = np.setdiff1d(self.current_indices, self.badvals)


    def __getitem__(self, i):
        assert(i < self.current_indices.shape[0])
        index = self.current_indices[i]
        cropped_image, mask, tumor, good, rectified = self.cropped(self.data[index,:,:,0], self.data[index,:,:,1], index)
        segment = self.data[index,:,:,1]
        tumor = self.tumor_transform(tumor)
        good = self.tumor_transform(good)
        if self.verticalflip and random.randint(0,1) == 1:
            target = torch.flipud(self.data[index,:,:,0]).unsqueeze(0)
            mask = torch.flipud(mask).unsqueeze(0)
            cropped_image = torch.flipud(cropped_image).unsqueeze(0)
            segment = torch.flipud(segment).unsqueeze(0)
            tumor = torch.flipud(tumor)
            good = torch.flipud(good)
        else:
            target = self.data[index,:,:,0].unsqueeze(0)
            mask = mask.unsqueeze(0)
            cropped_image = cropped_image.unsqueeze(0)
            segment = segment.unsqueeze(0)
        if self.loadRectifier:
            return self.transform(target), self.transform(mask), self.transform(cropped_image), self.transform(segment), tumor, good, rectified
        else:
            return self.transform(target), self.transform(mask), self.transform(cropped_image), self.transform(segment), tumor
        

    def display(self, i):
        assert(i < self.current_indices.shape[0])
        index = self.current_indices[i]
        a1 = self.data[index,:,:,0]
        a2 = self.data[index,:,:,1]
        c,_ = self.cropped(self.data[index,:,:,0], self.data[index,:,:,1], index)
        
        fig = plt.figure()
        plt.subplot(1,3,1)
        plt.imshow(a1, cmap = 'gray')
        plt.subplot(1,3,2)
        plt.imshow(a2, cmap = 'gray')
        plt.subplot(1,3,3)
        plt.imshow(c, cmap = 'gray')
        plt.show()

    def __len__(self):
        return self.current_indices.shape[0]