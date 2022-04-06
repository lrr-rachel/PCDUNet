from __future__ import print_function, division

import argparse
from tkinter import Variable
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from edvr_arch import PCDAlignment, EDVR
import torchvision
from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
import time
import os
import copy
import glob
from torch.utils.data import Dataset
from skimage import io, transform
from networks import UnetGenerator
import shutil
from PIL import Image
import gc
# from torch.autograd import Variable

gc.collect()

torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Main Unet')
parser.add_argument('--root_distorted', type=str, default='datasets/van_distorted', help='train and test datasets')
parser.add_argument('--root_restored', type=str, default='datasets/van_restored', help='save output images')
# parser.add_argument('--root_restored', type=str, default='', help='save output images')
parser.add_argument('--resultDir', type=str, default='results', help='save output images')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--numframes', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--cropsize', type=int, default=0)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--maxepoch', type=int, default=200)
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--deform', action='store_true', help='Run test only')
parser.add_argument('--network', type=str, default='EDVR')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--recon_network', type=str, default='resnet', help='For EDVR model')
parser.add_argument('--with_tsa', action='store_true', help='For EDVR model')
parser.add_argument('--with_predeblur', action='store_true', help='For EDVR model')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--num_feat',default=16, help='features for pcd')
args = parser.parse_args()


root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
cropsize = args.cropsize
savemodelname = args.savemodelname
maxepoch = args.maxepoch
NoNorm = args.NoNorm
deform = args.deform
network = args.network
retrain = args.retrain
recon_network = args.recon_network
with_tsa = args.with_tsa
with_predeblur = args.with_predeblur
topleft = args.topleft
num_feat = args.num_feat
# root_distorted='datasets/van_distorted'
# root_restored='datasets/van_restored'
# resultDir = 'results'
if not os.path.exists(resultDir):
    os.mkdir(resultDir)

class HeathazeDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_distorted, root_restored='', network='unet', numframes=3, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform
        if len(root_restored)==0:
            self.filesnames = glob.glob(os.path.join(root_distorted,'**_restored\*.png'))
        else:
            self.filesnames = glob.glob(os.path.join(root_restored,'*.png'))
        self.numframes = numframes
    def __len__(self):
        return len(self.filesnames)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        subname = self.filesnames[idx].split("\\")
        curf = int(subname[-1][:-4])
        halfcurf = int(self.numframes/2)
        if len(self.root_restored)==0:
            totalframes = len(glob.glob(os.path.join(os.path.dirname(os.path.abspath(self.filesnames[idx])), '*.png')))
        else:
            totalframes = len(self.filesnames)
        # print("TOTAL FRAMES:",totalframes)
        # print("[CURF]:",curf)
        # print("halfcurf: int(self.numframes/2) ->",halfcurf)
        # print(curf)

        if curf-halfcurf<=1:
            rangef = range(1,self.numframes+1)
        elif curf+halfcurf>=totalframes:
            if self.numframes==1:
                rangef = range(curf, curf+1)
            else:
                rangef = range(totalframes-self.numframes+1, totalframes+1)
        else:
            rangef = range(curf-halfcurf + 1 - (self.numframes % 2), curf+halfcurf+1)
        # print("rangef:",rangef)

        # print(rangef)
        dig = len(subname[-1])-4
        nameformat = '%0'+str(dig)+'d'
        # print("nameformat:",nameformat)
        # print('rangef '+str(rangef))
        for f in rangef:
            # read distorted image
            rootdistorted = os.path.join(os.path.dirname(os.path.abspath(self.filesnames[idx])),nameformat % f + ".png")
            rootdistorted = rootdistorted.replace('_restored', '_distorted')
            # print('Read Distorted: '+rootdistorted)
            temp = io.imread(rootdistorted)
            temp = temp.astype('float32')
            if network=='EDVR':
                temp = temp[..., np.newaxis]
            if f==rangef[0]:
                image = temp/255.
            else:
                if network=='EDVR':
                    image = np.append(image,temp/255.,axis=3)
                else:
                    image = np.append(image,temp/255.,axis=2)
        # read corresponding clean image
        # print("restored:",self.root_restored)

        if len(self.root_restored)==0:
            rootrestored = self.filesnames[idx]
            groundtruth = io.imread(rootrestored)
        else:
            if self.numframes == 1:
                groundtruth = io.imread(os.path.join(self.root_restored,subname[-1]))
            else:
                if curf-halfcurf<=1:
                    rootrestored = os.path.abspath(self.filesnames[self.numframes-1])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read GroundTruth:",rootrestored)
                elif curf+halfcurf>=totalframes:
                    rootrestored = os.path.abspath(self.filesnames[totalframes-1])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read GroundTruth:",rootrestored)
                else:
                    rootrestored = os.path.abspath(self.filesnames[(curf-halfcurf+1-(self.numframes%2))+self.numframes-2])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read GroundTruth:",rootrestored)

        groundtruth = groundtruth.astype('float32')
        groundtruth = groundtruth/255.
        sample = {'image': image, 'groundtruth': groundtruth}
        if self.transform:
            sample = self.transform(sample)
        return sample

class RandomCrop(object):
    def __init__(self, output_size, topleft=False):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        self.topleft = topleft
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        if (h > new_h) and (not topleft):
            top = np.random.randint(0, h - new_h)
        else:
            top = 0
        if (w > new_w) and (not topleft):
            left = np.random.randint(0, w - new_w)
        else:
            left = 0
        image = image[top: top + new_h,
                      left: left + new_w]
        groundtruth = groundtruth[top: top + new_h,
                      left: left + new_w]
        return {'image': image, 'groundtruth': groundtruth}

class ToTensor(object):
    def __init__(self, network='unet'):
        self.network = network
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        if network=='EDVR':
            image = image.transpose((3, 2, 0, 1))
        else:
            image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        if network=='EDVR':
            image = (image-0.5)/0.5
        else:
            vallist = [0.5]*image.shape[0]
            normmid = transforms.Normalize(vallist, vallist)
            image = normmid(image)
        # ground truth
        vallist = [0.5]*groundtruth.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        groundtruth = normmid(groundtruth)
        image = image.unsqueeze(0)
        groundtruth = groundtruth.unsqueeze(0)
        return {'image': image, 'groundtruth': groundtruth}

class RandomFlip(object):
    def __call__(self, sample):
        image, groundtruth = sample['image'], sample['groundtruth']
        op = np.random.randint(0, 3)
        if op<2:
            image = np.flip(image,op)
            groundtruth = np.flip(groundtruth,op)
        return {'image': image, 'groundtruth': groundtruth}

def readimage(filename, root_distorted, numframes=3, network='unet'):
    # read distorted image
    subname = filename.split("\\")
    curf = int(subname[-1][:-4])
    halfcurf = int(numframes/2)
    if curf==1:
        rangef = range(1,numframes+1)
    elif curf==len(filesnames):
        rangef = range(curf-int(numframes/2)-1, curf+1)
    else:
        rangef = range(curf-int(numframes/2), curf+int(numframes/2)+1)
    if curf-halfcurf<=1:
        rangef = range(1,numframes+1)
    elif curf+halfcurf>=len(filesnames):
        if numframes==1:
            rangef = range(curf, curf+1)
        else:
            rangef = range(len(filesnames)-numframes+1, len(filesnames)+1)
    else:
        rangef = range(curf-halfcurf, curf+halfcurf+1)
    dig = len(subname[-1])-4
    nameformat = '%0'+str(dig)+'d'
    for f in rangef:
        # read distorted image
        temp = io.imread(os.path.join(root_distorted,nameformat % f + ".png"))
        temp = temp.astype('float32')
        if network=='EDVR':
            temp = temp[..., np.newaxis]
        if f==rangef[0]:
            image = temp/255.
        else:
            if network=='EDVR':
                image = np.append(image,temp/255.,axis=3)
            else:
                image = np.append(image,temp/255.,axis=2)
    if network=='EDVR':
        image = image.transpose((3, 2, 0, 1))
    else:
        image = image.transpose((2, 0, 1))
    image = torch.from_numpy(image)
    if network=='EDVR':
        image = (image-0.5)/0.5
    else:
        vallist = [0.5]*image.shape[0]
        normmid = transforms.Normalize(vallist, vallist)
        image = normmid(image)
    image = image.unsqueeze(0)
    return image
# =====================================================================

# data loader
print("[INFO] Data Start Loading...")
if cropsize==0:
    heathaze_dataset = HeathazeDataset(root_distorted=root_distorted,
                                    root_restored=root_restored, network=network, numframes=numframes,
                                    transform=transforms.Compose([RandomFlip(),ToTensor(network=network)]))
else:
    heathaze_dataset = HeathazeDataset(root_distorted=root_distorted,
                                    root_restored=root_restored, network=network, numframes=numframes,
                                    transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("[INFO] Unet Generating...")
if network=='EDVR':
    model = EDVR(num_in_ch=3,num_out_ch=3,num_feat=num_feat,num_frame=numframes,deformable_groups=8,num_extract_block=5,
                 num_reconstruct_block=10,center_frame_idx=None,hr_in=True,with_predeblur=False,with_tsa=False,
                 num_downs= unetdepth)
else:
    model = UnetGenerator(input_nc=numframes*3, output_nc=3, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
if retrain:
    model.load_state_dict(torch.load(os.path.join(resultDir,savemodelname+'.pth.tar'),map_location=device))
model = model.to(device)

criterion = nn.MSELoss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# =====================================================================
print("[INFO] Start Training...")
num_epochs=maxepoch
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 100000000.0
for epoch in range(num_epochs+1):
    # print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    # Each epoch has a training and validation phase
    for phase in ['train']:#, 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode
        running_loss = 0.0
        running_corrects = 0
        # Iterate over data.
        for i in range(len(heathaze_dataset)):
            sample = heathaze_dataset[i]
            inputs = sample['image'].to(device)
            labels = sample['groundtruth'].to(device)
            # inputs = Variable(inputs,requires_grad=True)
            # labels = Variable(labels,requires_grad=True)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with torch.no_grad():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                # loss = Variable(loss,requires_grad=True)
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(heathaze_dataset)
        # print('\n')
        print('[Epoch] ' + str(epoch),':' + '[Loss] ' + str(epoch_loss))
        # print('\n')
        if (epoch % 50) == 0:
            torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))
        # deep copy the model
        if (epoch>10) and (epoch_loss < best_acc):
            best_acc = epoch_loss
            torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))

# =======TESTING==============================================================
resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)
if network=='EDVR':
    model = EDVR(num_in_ch=3,num_out_ch=3,num_feat=num_feat,num_frame=numframes,deformable_groups=8,num_extract_block=5,
                 num_reconstruct_block=10,center_frame_idx=None,hr_in=True,with_predeblur=False,with_tsa=False,
                 num_downs= unetdepth)
else:
    model = UnetGenerator(input_nc=numframes*3, output_nc=3, num_downs=unetdepth, deformable=deform, norm_layer=NoNorm)
model.load_state_dict(torch.load(os.path.join(resultDir,'best_'+savemodelname+'.pth.tar'),map_location=device))
model.eval()
model = model.to(device)

# =====================================================================
filesnames = glob.glob(os.path.join(root_distorted,'*.png'))
for i in range(len(filesnames)):
    curfile = filesnames[i]
    inputs = readimage(curfile, root_distorted, numframes, network=network)
    subname = curfile.split("\\")
    inputs = inputs.to(device)
    with torch.no_grad():
        output = model(inputs)
        output = output.squeeze(0)
        output = output.cpu().numpy()
        output = output.transpose((1, 2, 0))
        output = (output*0.5 + 0.5)*255
        io.imsave(os.path.join(resultDirOutImg, subname[-1]), output.astype(np.uint8))
