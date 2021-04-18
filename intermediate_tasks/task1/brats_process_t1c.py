import numpy as np
import torch
import random
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch import nn, optim
from torch.nn import functional as F
import PIL
import matplotlib.pyplot as plt
import os
import sys
import random
import time
import argparse
import re
from tqdm import tqdm
from medpy.io.load import load as loadmed

track = {}
ids_assigned = {}
newid = 0
data = torch.zeros((274,2,240,240,26))

inv_transform = transforms.Compose([
                        transforms.ToPILImage(),
                        transforms.Resize(256),
                        transforms.ToTensor(),
                                ])

data_path = '/content/drive/Shareddrives/Datasets/BRATS/'
for root, dirs, files in tqdm(os.walk(os.path.join(data_path, 'train'), topdown=True), desc = 'Image Number',total = 277):
    for name in files:
        path = os.path.join(root, name)
        if not (('T1c.' in name) or ('OT.' in name)):
            continue
        dir_split = path.split('/')
        patient_code = dir_split[-2]
        patient_type = dir_split[-3]
        patient_code = patient_type + patient_code
        if patient_code in track.keys():
            track[patient_code] += 1
        else:
            track[patient_code] = 1
        im = loadmed(path)[0][:,:,70:96]
        imsum = np.expand_dims(np.expand_dims(im.max(0).max(0),0),0)
        im = im/imsum
        assert(im.shape[0] == 240 and im.shape[1] == 240)

        if patient_code in ids_assigned.keys():
            im_id = ids_assigned[patient_code]
        else:
            im_id = newid
            ids_assigned[patient_code] = newid
            newid += 1
        if 'T1c.' in name:
            data[im_id,0,:,:,:] = torch.FloatTensor(im)
        else:
            data[im_id,1,:,:,:] = torch.FloatTensor(im)


torch.save({'data':data},os.path.join(data_path, 'data_t1c_ot.pth'))
a = np.array(list(track.values()))
k = np.array(list(track.keys()))
assert((a==2).sum() == 274)