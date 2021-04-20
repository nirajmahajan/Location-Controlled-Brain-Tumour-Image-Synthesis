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

from myModels import ContextEncoder
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
SAVE_INTERVAL = 25
STATE_INTERVAL = 200
NUM_EPOCHS = 2000
EXPERIMENT_ID = args.id
expected_id = '3.1'
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

dataset = BRATS_T1c(train_ratio = 0.75, transform = transform, verticalflip = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, num_workers = 2, shuffle = True)
dataset.set_train(train = True)
dataset.set_healthy(healthy = True)

rectifier = ContextEncoder()

global pre_e
pre_e = 0

if args.resume or args.resume_from_drive:
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
    rectifier.load_state_dict(dic['rectifier'])
    rectifier.optimizer_G.load_state_dict(dic['optimizer_G'])
    rectifier.optimizer_D.load_state_dict(dic['optimizer_D'])
    glosses = dic['glosses']
    dlosses = dic['dlosses']
    print('Resuming Training from epoch {} for Experiment {}'.format(pre_e,EXPERIMENT_ID))
else:
    model_state = 0
    pre_e =0
    print('Starting Training for Experiment {}'.format(EXPERIMENT_ID))

def is_eval_mode():
    return args.eval

def train(e):
    print('\nTraining for epoch {}'.format(e))
    tot_loss_g = 0
    tot_loss_d = 0

    for batch_num,(target,mask,cropped,segment,tumor) in tqdm(enumerate(dataloader), desc = 'Epoch {}'.format(e), total = len(dataloader)):
        lg, ld = rectifier.train_epoch(e, batch_num, cropped, target)
        tot_loss_d += ld
        tot_loss_g += lg

    print('Total Generator Loss for epoch = {}'.format(tot_loss_g/batch_num))
    print('Total Discriminator Loss for epoch = {}'.format(tot_loss_d/batch_num))
    return tot_loss_g/batch_num, tot_loss_d/batch_num

def validate():
    print('\nTesting')
    dataset.set_train(train = False)
    dataset.set_healthy(healthy = None)
    num_samples = TRAIN_BATCH_SIZE

    with torch.no_grad():
        for batch_num,(target,mask,cropped,segment,tumor) in enumerate(dataloader):
            rectifier.eval()
            fake = rectifier.generate(cropped.to(device))
            break
        for i in range(num_samples):
            s = cropped[i].squeeze().detach().cpu().numpy()
            t = target[i].squeeze().detach().cpu().numpy()
            f = fake[i].squeeze().detach().cpu().numpy()
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
            plt.savefig('images/test/test_{}.png'.format(i))

    os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
    os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))
    

if args.eval:
    validate()
    os._exit(0)

for e in range(NUM_EPOCHS):

    model_state = e//STATE_INTERVAL
    if pre_e > 0:
        pre_e -= 1
        continue

    if e % SAVE_INTERVAL == 0:
        seed_torch(args.seed)

    lg, ld = train(e)
    glosses.append(lg)
    dlosses.append(ld)

    dic = {}
    dic['e'] = e+1
    dic['rectifier'] = rectifier.state_dict()
    dic['optimizer_G'] = rectifier.optimizer_G.state_dict()
    dic['optimizer_D'] = rectifier.optimizer_D.state_dict()
    dic['dlosses'] = dlosses
    dic['glosses'] = glosses


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, local_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, local_path + 'state.pth')
        print('Saving model to {}'.format(runtime_path))
        print('Copying checkpoint to drive')
        os.system("cp -r ./models/checkpoint_{}.pth '{}'".format(model_state, runtime_path))
        os.system("cp -r ./models/state.pth '{}'".format(runtime_path))
        os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
        os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))