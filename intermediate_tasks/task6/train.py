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
NUM_EPOCHS = 500
EXPERIMENT_ID = args.id
expected_id = '6'
assert(expected_id == EXPERIMENT_ID)
losses = []
accuracies = []
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
            transforms.Resize((256,256)),
                    ])

dataset = BRATS_T1c(train_ratio = 0.75, transform = transform, verticalflip = True, loadRectifier = True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=TRAIN_BATCH_SIZE, shuffle = True)
dataset.set_train(train = True)
dataset.set_healthy(healthy = False)

model = models.AlexNet(num_classes = 2).to(device)
model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2)).to(device)
criterion = nn.CrossEntropyLoss().to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=3e-3, momentum = 0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

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
    model.load_state_dict(dic['model'])
    optimizer.load_state_dict(dic['optimizer'])
    scheduler.load_state_dict(dic['scheduler'])
    losses = dic['losses']
    accuracies = dic['accuracies']
    print('Resuming Training from epoch {} for Experiment {}'.format(pre_e,EXPERIMENT_ID))
else:
    model_state = 0
    pre_e =0
    print('Starting Training for Experiment {}'.format(EXPERIMENT_ID))

def is_eval_mode():
    return args.eval

def train(e):
    dataset.set_train(True)
    dataset.set_healthy(False)
    # print('\nTraining for epoch {}'.format(e))
    tot_loss = 0
    tot_correct = 0
    tot = 0

    for batch_num,(target,mask,cropped,segment,tumor,good,rectified) in tqdm(enumerate(dataloader), desc = 'Epoch {}'.format(e), total = len(dataloader)):
        optimizer.zero_grad()
        data = torch.cat((good,tumor), 0).to(device)
        labels = torch.ones(data.shape[0]).type(torch.int64).to(device)
        labels[:good.shape[0]] = 0
        shuffler = np.arange(labels.shape[0])
        np.random.shuffle(shuffler)
        labels = labels[shuffler]
        data = data[shuffler]
        
        model.train()
        outp = model(data)
        loss = criterion(outp, labels)
        loss.backward()
        optimizer.step()

        preds = outp.max(1)[1]
        tot_correct += (preds.detach().cpu().numpy() == labels.cpu().numpy()).sum()
        tot += preds.shape[0]
        tot_loss += loss.item()

    print('Total Loss for epoch = {}'.format(tot_loss/batch_num))
    print('Train Accuracy for epoch = {}'.format(100*tot_correct/tot))
    return 100*tot_loss/batch_num

def validate(silent = True):
    if silent:
        print('\nTesting')
    dataset.set_train(False)
    dataset.set_healthy(False)
    tot_correct = 0
    tot = 0

    with torch.no_grad():
        for batch_num,(target,mask,cropped,segment,tumor,good,rectified) in tqdm(enumerate(dataloader), desc = 'Epoch {}'.format(e), total = len(dataloader)):
            data = torch.cat((good,tumor), 0).to(device)
            labels = torch.ones(data.shape[0]).type(torch.int64).to(device)
            labels[:good.shape[0]] = 0
            
            model.eval()
            outp = model(data)

            preds = outp.max(1)[1]
            tot_correct += (preds.detach().cpu().numpy() == labels.cpu().numpy()).sum()
            tot += preds.shape[0]

        print('Test Accuracy for epoch = {}\n'.format(100*tot_correct/tot))
        return 100*tot_correct/tot
    

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

    l = train(e)
    losses.append(l)
    a = validate()
    accuracies.append(a)

    dic = {}
    dic['e'] = e+1
    dic['model'] = model.state_dict()
    dic['optimizer'] = optimizer.state_dict()
    dic['scheduler'] = scheduler.state_dict()
    dic['losses'] = losses
    dic['accuracies'] = accuracies


    if (e+1) % SAVE_INTERVAL == 0:
        torch.save(dic, local_path + 'checkpoint_{}.pth'.format(model_state))
        torch.save({'state': model_state}, local_path + 'state.pth')
        print('Saving model to {}'.format(runtime_path))
        print('Copying checkpoint to drive')
        os.system("cp -r ./models/checkpoint_{}.pth '{}'".format(model_state, runtime_path))
        os.system("cp -r ./models/state.pth '{}'".format(runtime_path))
        os.system("cp -r ./images/train/* '{}'".format(os.path.join(runtime_path,'./images/train/')))
        os.system("cp -r ./images/test/* '{}'".format(os.path.join(runtime_path,'./images/test/')))