import numpy as np
import torch
import random
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms, models, datasets
from torch import nn, optim
from torch.nn import functional as F
import pickle
import argparse
import sys
import os
import PIL
import time
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import gc
import matplotlib.patches as patches

from myModels import Pix2pix
from myDatasets import BRATS_T1c

def seed_torch(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


parser = argparse.ArgumentParser()
parser.add_argument('--resume', action = 'store_true')
parser.add_argument('--resume_from_drive', action = 'store_true')
parser.add_argument('--eval', action = 'store_true')
parser.add_argument('--seed', type = int, default = 0)
parser.add_argument('--state', type = int, default = -1)
parser.add_argument('--id', type = str, default = '1.1')
args = parser.parse_args()

seed_torch(args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TRAIN_BATCH_SIZE = 30
SAVE_INTERVAL = 5
STATE_INTERVAL = 200
NUM_EPOCHS = 2000
EXPERIMENT_ID = args.id
expected_id = '5.2'
assert(expected_id == EXPERIMENT_ID)
glosses = []
dlosses = []
torch.autograd.set_detect_anomaly(True)


runtime_path = '/content/drive/Shareddrives/MIC_Project_abhinav_niraj/tasks/{}/'.format(EXPERIMENT_ID)
if not os.path.isdir(runtime_path):
    os.mkdir(runtime_path)
local_path = './models/'
if not os.path.isdir(local_path):
    os.mkdir(local_path)
if not os.path.isdir('./images/'):
    os.mkdir('./images')
if not os.path.isdir('./images/train'):
    os.mkdir('./images/train')
if not os.path.isdir('./images/test'):
    os.mkdir('./images/test')
if not os.path.isdir(os.path.join(runtime_path,'./images/')):
    os.mkdir(os.path.join(runtime_path,'./images/'))
if not os.path.isdir(os.path.join(runtime_path,'./images/train')):
    os.mkdir(os.path.join(runtime_path,'./images/train'))
if not os.path.isdir(os.path.join(runtime_path,'./images/test')):
    os.mkdir(os.path.join(runtime_path,'./images/test'))

transform = transforms.Compose([
            transforms.Resize(256),
                    ])

dataset = BRATS_T1c(train_ratio = 0.75, transform = transform, verticalflip = True, loadRectifier = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle = True)
dataset.set_train(train = True)
dataset.set_healthy(healthy = False)

model = Pix2pix()

global pre_e
pre_e = 0

if args.resume or args.resume_from_drive or 1:
    if args.resume_from_drive:
        print('Copying the checkpoint to the runtime')
        os.system("cp -r '{}'state.pth* ./models".format(runtime_path))
        model_state = torch.load(local_path + 'state.pth')['state']
        os.system("cp -r '{}'checkpoint_{}.pth ./models".format(runtime_path, model_state))
    model_state = torch.load(local_path + 'state.pth')['state']
    if (not args.state == -1):
        model_state = args.state
    print('Loading checkpoint at model state {}'.format(model_state))
    dic = torch.load(local_path + 'checkpoint_{}.pth'.format(model_state))
    pre_e = dic['e']
    model.load_state_dict(dic['model'])
    model.optimizer_G.load_state_dict(dic['optimizer_G'])
    model.optimizer_D.load_state_dict(dic['optimizer_D'])
    glosses = dic['glosses']
    dlosses = dic['dlosses']
    print('Resuming Training from epoch {} for Experiment {}'.format(pre_e,EXPERIMENT_ID))
else:
    model_state = 0
    pre_e =0
    print('Starting Training for Experiment {}'.format(EXPERIMENT_ID))

def is_eval_mode():
    return args.eval

def cropp(a2):
    rr = torch.arange(a2.shape[1])[(a2>0.5).sum(1) > 0]
    cc = torch.arange(a2.shape[0])[(a2>0.5).sum(0) > 0]
    rmin = rr.min()
    rmax = rr.max()
    cmin = cc.min()
    cmax = cc.max()
    return rmin, rmax, cmin, cmax

def train(e):
    print('\nTraining for epoch {}'.format(e))
    tot_loss_g = 0
    tot_loss_d = 0

    for batch_num,(target,mask,cropped,segment,tumor,good_cropped,rectified) in tqdm(enumerate(dataloader), desc = 'Epoch {}'.format(e), total = len(dataloader)):
        lg, ld = model.train_epoch(e, batch_num, mask,rectified, target)
        tot_loss_d += ld
        tot_loss_g += lg

    print('Total Generator Loss for epoch = {}'.format(tot_loss_g/batch_num))
    print('Total Discriminator Loss for epoch = {}'.format(tot_loss_d/batch_num))
    return tot_loss_g/batch_num, tot_loss_d/batch_num

masklist = [torch.zeros((30,1,256,256)) for i in range(4)]
masklist[0][:,0,70:115,100:150] = 1
masklist[1][:,0,120:150,120:144] = 1
masklist[2][:,0,90:120,160:200] = 1
masklist[3][:,0,135:170,100:150] = 1

fakelists = []

def validate():
    print('\nTesting')
    dataset.set_train(train = False)
    dataset.set_healthy(healthy = False)
    num_samples = TRAIN_BATCH_SIZE

    with torch.no_grad():
        for batch_num,(target,mask,cropped,segment,tumor,good_cropped,rectified) in enumerate(dataloader):
            model.eval()
            for ii in range(4):
                fakelists.append(model.generate(masklist[ii],rectified))
            break
        for i in range(num_samples):
            for ii in range(4):
                r = (rectified[i]-rectified[i].min())
                r = r/r.max()
                t = PIL.Image.fromarray((255*r).squeeze().detach().cpu().numpy().astype(np.uint8))
                rmin,rmax,cmin,cmax = cropp(masklist[ii][0,0,:,:].squeeze())
                rect = patches.Rectangle((cmin,rmin), (cmax-cmin), (rmax-rmin), linewidth=1, edgecolor='r', facecolor='none')
                rect2 = patches.Rectangle((cmin,rmin), (cmax-cmin), (rmax-rmin), linewidth=1, edgecolor='r', facecolor='none')
                f = fakelists[ii][i]
                f = f - f.min()
                f = f/f.max()
                f = PIL.Image.fromarray((255*f).squeeze().detach().cpu().numpy().astype(np.uint8))
                fig, ax = plt.subplots(1,2,figsize = (6,3), sharex = True)
                ax[0].imshow(t, cmap = 'gray')
                ax[0].set_title('Healthy Image')
                ax[0].add_patch(rect)
                ax[0].axis('off')
                ax[1].imshow(f, cmap = 'gray')
                ax[1].set_title('Generated Image')
                ax[1].add_patch(rect2)
                ax[1].axis('off')
                plt.savefig('images/test/test_{}_{}.png'.format(i, ii))

    os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
    os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))
    

validate()
 