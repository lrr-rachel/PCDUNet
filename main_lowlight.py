
from __future__ import print_function, division

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from edvr_arch import PCDAlignment, PCDUNet
import torchvision
from torchvision import datasets, models, transforms
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
import matplotlib.pyplot as plt
from math import log10, sqrt
import cv2
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio 
from skimage.metrics import structural_similarity 

gc.collect()
torch.cuda.empty_cache()
parser = argparse.ArgumentParser(description='Main Unet')
parser.add_argument('--root_distorted', type=str, default='/user/home/gf19473/scratch/lowlight_dataset/input/', help='train and test datasets')
parser.add_argument('--root_restored', type=str, default='/user/home/gf19473/scratch/lowlight_dataset/gt/', help='save output images')
parser.add_argument('--root_test', type=str, default='/user/home/gf19473/scratch/lowlight_dataset/input/', help='save output images')

# parser.add_argument('--root_restored', type=str, default='', help='save output images')
parser.add_argument('--resultDir', type=str, default='pcdunet_frame5', help='save output images')
parser.add_argument('--unetdepth', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--numframes', type=int, default=5, metavar='N',  help='number of epochs to train (default: 10)')
parser.add_argument('--cropsize', type=int, default=512)
parser.add_argument('--savemodelname', type=str, default='model')
parser.add_argument('--maxepoch', type=int, default=10)
parser.add_argument('--NoNorm', action='store_false', help='Run test only')
parser.add_argument('--network', type=str, default='PCDUNet')
parser.add_argument('--retrain', action='store_true')
parser.add_argument('--recon_network', type=str, default='resnet', help='For PCDUNet model')
parser.add_argument('--topleft', action='store_true', help='crop using top left')
parser.add_argument('--num_feat',default=16, help='features for pcd')
parser.add_argument('--readgtr',type=str,default='middleframe', help='method for reading groundtruth')
parser.add_argument('--lowlightmode', action='store_true', help='using lowlight dataset')

parser.add_argument('--lossgraph',default='false', help='loss function graph')
parser.add_argument('--resize_height',type=int,default=512,help='resize for modifying unetdepth')
parser.add_argument('--resize_width',type=int,default=512, help='resize for modifying unetdepth')

# inputtype: Frame or Video
parser.add_argument('--inputtype',default='Frame', help='for Video Stream Input')
parser.add_argument('--videoDirIn',default='./datasets/distorted.mp4/', help='Video Stream Processing')
parser.add_argument('--videoDirOut',default='results/video.avi',help='Video Stream Processing')
args = parser.parse_args()

if torch.cuda.is_available():
    print('using GPU')
else:
    print('using CPU')


root_distorted = args.root_distorted
root_restored  = args.root_restored
resultDir = args.resultDir
unetdepth = args.unetdepth
numframes = args.numframes
cropsize = args.cropsize
savemodelname = args.savemodelname
maxepoch = args.maxepoch
NoNorm = args.NoNorm
network = args.network
retrain = args.retrain
recon_network = args.recon_network
topleft = args.topleft
num_feat = args.num_feat
readgtr = args.readgtr
lossgraph = args.lossgraph
resize_height = args.resize_height
resize_width = args.resize_width
inputtype = args.inputtype
videoDirIn = args.videoDirIn
videoDirOut = args.videoDirOut
lowlightmode = args.lowlightmode

if not os.path.exists(resultDir):
    os.mkdir(resultDir)

train_txt_file = '/user/home/bv18502/scratch/video_enhance/datasets/4_vilab/train_list.txt'
test_txt_file = '/user/home/bv18502/scratch/video_enhance/datasets/4_vilab/test_list.txt'

class LowLightDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_distorted, root_restored='', network='unet', numframes=3, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform

        # Read folder names from the training txt file
        with open(train_txt_file, 'r') as file:
            self.folder_names = file.read().splitlines()
        print("read training folder names: ", self.folder_names)

        self.filesnames = []  # Restored directory names
        self.distortednames = []  # Distorted directory names

        for folder_name in self.folder_names:
            data_folder_distorted = os.path.join(self.root_distorted, folder_name)
            data_folder_restored = os.path.join(self.root_restored, folder_name)
            # print("Data folder distorted:", data_folder_distorted)
            # print("Data folder restored:", data_folder_restored)

            self.filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_10', '*.png')))
            self.filesnames.extend(glob.glob(os.path.join(data_folder_restored, 'normal_light_20', '*.png')))
            self.distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_10', '*.png')))
            self.distortednames.extend(glob.glob(os.path.join(data_folder_distorted, 'low_light_20', '*.png')))

        print("Length restored = ", len(self.filesnames))
        print("Length distorted = ", len(self.distortednames))
        self.numframes = numframes


    def __len__(self):
        return len(self.filesnames)


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # subname = self.filesnames[idx].split("\\")
        # curf = int(subname[-1][:-4])
        curf = idx+1

        halfcurf = int(self.numframes/2)
        totalframes = len(self.filesnames)
        # print("TOTAL FRAMES:",totalframes)
        # print("[CURF]:",curf)
        # print("halfcurf=numframes/2: ",halfcurf)
        
        if curf-halfcurf<=1:
            # close to the beginning of the sequence
            rangef = range(1,self.numframes+1)
        elif curf+halfcurf>=totalframes:
            # close to the end of the sequence
            if self.numframes==1:
                rangef = range(curf, curf+1)
            else:
                rangef = range(totalframes-self.numframes+1, totalframes+1)
        else:
            rangef = range(curf-halfcurf + 1 - (self.numframes % 2), curf+halfcurf+1)

        # dig = len(subname[-1])-4
        # nameformat = '%0'+str(dig)+'d'
        # print("Rangef: ", rangef)

        for f in rangef:
            rootdistorted = self.distortednames[f-1]

            # print('Read Distorted: '+rootdistorted)

            temp = cv2.imread(rootdistorted, cv2.IMREAD_COLOR)
            # temp = cv2.resize(temp, (resize_width,resize_height))
            temp = temp.astype('float32')

            if network=='PCDUNet':
                temp = temp[..., np.newaxis]
            if f==rangef[0]:
                image = temp/255.
            else:
                if network=='PCDUNet':
                    image = np.append(image,temp/255.,axis=3)
                else:
                    image = np.append(image,temp/255.,axis=2)

        rootrestored = self.filesnames[idx]
        groundtruth = cv2.imread(rootrestored, cv2.IMREAD_COLOR)
        # groundtruth = cv2.resize(groundtruth, (resize_width,resize_height),cv2.INTER_CUBIC)
        # print("Read Restored:",rootrestored)
        groundtruth = groundtruth.astype('float32')


        groundtruth = groundtruth/255.
        sample = {'image': image, 'groundtruth': groundtruth}
        if self.transform:
            sample = self.transform(sample)
        return sample

class HeathazeDataset(Dataset):
    """Face Landmarks dataset."""
    def __init__(self, root_distorted, root_restored='', network='unet', numframes=3, transform=None):
        self.root_distorted = root_distorted
        self.root_restored = root_restored
        self.transform = transform
        if len(root_restored)==0:
            self.filesnames = glob.glob(os.path.join(root_distorted,'**_restored/*.png'))
        else:
            print(os.path.join(root_restored,'*.png'))
            self.filesnames = glob.glob(os.path.join(root_restored,'*.png'))
        # print(len(self.filesnames))
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
        #print("rangef:",rangef)

        dig = len(subname[-1])-4
        nameformat = '%0'+str(dig)+'d'
        #print(nameformat)

        for f in rangef:
            # read distorted image
            rootdistorted = os.path.join(os.path.dirname(os.path.abspath(self.filesnames[idx])),nameformat % f + ".png")
            rootdistorted = rootdistorted.replace('_restored', '_distorted')
            # print('Read Distorted: '+rootdistorted)
            temp = io.imread(rootdistorted)
            temp = temp.astype('float32')
            if network=='PCDUNet':
                temp = temp[..., np.newaxis]
            if f==rangef[0]:
                image = temp/255.
            else:
                if network=='PCDUNet':
                    image = np.append(image,temp/255.,axis=3)
                else:
                    image = np.append(image,temp/255.,axis=2)
        # read corresponding clean image
        # print("restored:",self.root_restored)

        if len(self.root_restored)==0:
            rootrestored = self.filesnames[idx]
            groundtruth = io.imread(rootrestored)
        else:
            if self.numframes >= 1:
                # print('read gt')
                groundtruth = io.imread(os.path.join(self.root_restored,subname[-1]))
                #print('groundtruth ' + subname[-1])
            elif readgtr == 'firstframe':
                if curf-halfcurf<=1:
                    rootrestored = os.path.abspath(self.filesnames[self.numframes-1])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read first GroundTruth:",rootrestored)
                elif curf+halfcurf>=totalframes:
                    rootrestored = os.path.abspath(self.filesnames[totalframes-1])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read first GroundTruth:",rootrestored)
                else:
                    gt = (curf-halfcurf+1-(self.numframes%2)) + self.numframes-2
                    rootrestored = os.path.abspath(self.filesnames[gt])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read first GroundTruth:",rootrestored)
            elif readgtr == 'middleframe':
                if curf-halfcurf<=1:
                    rootrestored = os.path.abspath(self.filesnames[halfcurf])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read middle GroundTruth:",rootrestored)
                elif curf+halfcurf>=totalframes:
                    rootrestored = os.path.abspath(self.filesnames[totalframes-halfcurf-1])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read middle GroundTruth:",rootrestored)
                else:
                    gt = (curf-halfcurf+1-(self.numframes%2)) + halfcurf - 1
                    rootrestored = os.path.abspath(self.filesnames[gt])
                    groundtruth = io.imread(rootrestored)
                    # print("[input: multiple] Read middle GroundTruth:",rootrestored)

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
        if network=='PCDUNet':
            image = image.transpose((3, 2, 0, 1))
        else:
            image = image.transpose((2, 0, 1))
        groundtruth = groundtruth.transpose((2, 0, 1))
        image = torch.from_numpy(image.copy())
        groundtruth = torch.from_numpy(groundtruth.copy())
        # image
        if network=='PCDUNet':
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

# x-t direction rippled image
def y_append(path, num, height):
    '''num: total number of frames
       height: image height
    '''
    dirs = os.listdir(path)
    output_image = np.zeros((height,num,3), np.uint8)
    indout = 0
    for item in dirs:
        if os.path.isfile(path+item):
            img = cv2.imread(path+item)
            column_y = img[:,100]
            output_image[:,indout,:] = column_y
            indout = indout + 1
    cv2.imshow('x-t image',output_image)
    cv2.imwrite('x-t.jpg', output_image)
    cv2.waitKey(0)  

# y-t direction rippled image
def x_append(path, num, width):
    '''num: total number of frames
       width: image width
    '''
    dirs = os.listdir(path)
    output_image = np.zeros((num,width,3), np.uint8)
    indout = 0
    for item in dirs:
        if os.path.isfile(path+item):
            img = cv2.imread(path+item)
            row_x = img[250,:]
            output_image[indout,:,:] = row_x
            indout = indout + 1
    cv2.imshow('y-t image',output_image)
    cv2.imwrite('y-t.jpg', output_image)
    cv2.waitKey(0)

def resizeimage(root_distorted, root_restored, width, height):
    print("[INFO] resizing distorted images")
    dirs_dis = os.listdir(root_distorted)
    for item in dirs_dis:
        if os.path.isfile(root_distorted+'/'+item):
            im = Image.open(root_distorted+'/'+item)
            f, e = os.path.splitext(root_distorted+'/'+item)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)
    print("[INFO] resizing restored images")
    dirs_res = os.listdir(root_restored)
    for item in dirs_res:
        if os.path.isfile(root_restored+'/'+item):
            im = Image.open(root_restored+'/'+item)
            f, e = os.path.splitext(root_restored+'/'+item)
            imResize = im.resize((width,height), Image.ANTIALIAS)
            imResize.save(f + '.png', 'PNG', quality=90)

def video_to_frames(input_loc, output_loc):
    """Extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc - Input video file.
        output_loc - Output directory to save the frames.
    """

    if not os.path.exists(output_loc):
        os.mkdir(output_loc)
    # Log the time
    time_start = time.time()
    # Start capturing the feed
    cap = cv2.VideoCapture(input_loc)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    while cap.isOpened():
        # Extract the frame
        ret, frame = cap.read()
        # skip malformed frames if present
        if not ret:  
            continue
        # Write the results back to output location.
        cv2.imwrite(output_loc + "/%#05d.jpg" % (count+1), frame)
        count = count + 1
        # If there are no more frames left
        if (count > (video_length-1)):
            # Log the time again
            time_end = time.time()
            # Release the feed
            cap.release()
            # Print stats
            print ("Done extracting frames.\n%d frames extracted" % count)
            print ("It took %d seconds for conversion." % (time_end-time_start))
            break

def frames_to_video(pathIn,pathOut,fps):
    '''Generate video stream file from input frames 
   
    Args:
        pathIn - Input path of result frames
        pathOut - Output path of generated video stream
        fps - Framerate of the created video stream
    '''

    frame_array = []
    dirs = os.listdir(pathIn)
    # print(dirs)
    for i in dirs:
        if os.path.isfile(pathIn+i):
            # Log the time
            time_start = time.time()
            filename=pathIn+i
            # print(filename)
            #reading each files
            img = cv2.imread(filename)
            (height, width, layers) = img.shape
            size = (width,height)
            # print(filename)
            #inserting the frames into an image array
            frame_array.append(img)

    # Initialize the video writer and save the frames to a video file
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    # Log the time again
    time_end = time.time()
    print ("Done converting frames to video stream")
    print ("It took %d seconds for conversion." % (time_end-time_start))
    # Release the VideoWriter
    out.release()

# =====================================================================
# converting video stream to frames
if inputtype == 'Video':
    input_loc = videoDirIn
    output_loc = root_distorted
    print("[INFO] Converting Video Stream to Frames...")
    video_to_frames(input_loc, output_loc)

if unetdepth > 5:
    print("[INFO] Data Preparation - Resizing")
    # resize restored and distorted images when modifying unet depth
    # image size must be divisible by 2^(unetdepth)
    # e.g., image size of 512x384
    resizeimage(root_distorted,root_restored,resize_width,resize_height)


# data loader
print("[INFO] Data Start Loading...")
if lowlightmode:
    print("Low Light Dataset")
    heathaze_dataset = LowLightDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))
else:
    if cropsize==0:
        heathaze_dataset = HeathazeDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomFlip(),ToTensor(network=network)]))
    else:
        heathaze_dataset = HeathazeDataset(root_distorted=root_distorted,
                                        root_restored=root_restored, network=network, numframes=numframes,
                                        transform=transforms.Compose([RandomCrop(cropsize, topleft=topleft),RandomFlip(),ToTensor(network=network)]))

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Check for the availability of GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print(f"Number of available GPUs: {torch.cuda.device_count()}")

# batch_size = 32
# data_loader = DataLoader(heathaze_dataset, batch_size=batch_size)


resultDirOutImg = os.path.join(resultDir,savemodelname)
if not os.path.exists(resultDirOutImg):
    os.mkdir(resultDirOutImg)

resultDirOut = os.path.join(resultDir,"trainingImg")
if not os.path.exists(resultDirOut):
    os.mkdir(resultDirOut)

print("[INFO] Network Generating...")
if network=='PCDUNet':
    print('PCDUNet')
    # changed hr_in to False (input size divided by 4)
    model = PCDUNet(num_in_ch=3,num_out_ch=3,num_feat=num_feat,num_frame=numframes,deformable_groups=8,num_extract_block=5,
                    num_reconstruct_block=10,center_frame_idx=None,hr_in=True, num_downs= unetdepth, norm_layer=NoNorm)
else:
    print('Unet')
    model = UnetGenerator(input_nc=numframes*3, output_nc=3, num_downs=unetdepth, norm_layer=NoNorm)
if retrain:
    model.load_state_dict(torch.load(os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'),map_location=device))
model = model.to(device)
# use all GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

# criterion = nn.MSELoss()
criterion = nn.L1Loss()

# Observe that all parameters are being optimized
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# =====================================================================
print("[INFO] Start Training...")
# Log the time
time_start = time.time()
num_epochs=maxepoch
since = time.time()
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 100000000.0
train_loss = []
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
        # Iterate over data
        
        # for batch in data_loader:
        #     inputs = batch['image'].to(device)
        #     labels = batch['groundtruth'].to(device)
        for i in range(len(heathaze_dataset)):
            sample = heathaze_dataset[i]
            inputs = sample['image'].to(device)
            labels = sample['groundtruth'].to(device)

            # print('input ' + str(inputs.shape))
            # print('labels ' + str(labels.shape))
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
            # with torch.no_grad():
                outputs = model(inputs)
                # if (i == 10) and ((epoch % 50) == 0):
                if (i == 10):
                    output = outputs.clone()
                    output = output.squeeze(0)
                    output = output.detach().cpu().numpy()
                    output = output.transpose((1, 2, 0))
                    output = (output*0.5 + 0.5)*255
                    io.imsave(os.path.join(resultDirOut, 'training'+ str(epoch) + '.png'), output.astype(np.uint8))
                    #output = labels.clone()
                    #output = output.squeeze(0)
                    #output = output.detach().cpu().numpy()
                    #output = output.transpose((1, 2, 0))
                    #output = (output*0.5 + 0.5)*255
                    #io.imsave(os.path.join(resultDirOutImg, 'input.png'), output.astype(np.uint8))
                loss = criterion(outputs, labels)
                #loss.requires_grad_(True)
                loss.backward()
                optimizer.step()
            # statistics
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(heathaze_dataset)
        if lossgraph == 'true':
            train_loss.append(epoch_loss)
            
        # print('\n')
        print('[Epoch] ' + str(epoch),':' + '[Loss] ' + str(epoch_loss))
        # print('\n')
        # if (epoch % 20) == 0:
        torch.save(model.state_dict(), os.path.join(resultDir, savemodelname + '_ep'+str(epoch)+'.pth.tar'))
        np.save(os.path.join(resultDir,'loss_array.npy'), np.array(train_loss))
        # deep copy the model
        if (epoch>=3) and (epoch_loss < best_acc):
            best_acc = epoch_loss
            torch.save(model.state_dict(), os.path.join(resultDir, 'best_'+savemodelname+'.pth.tar'))
            
# Log the time again
time_end = time.time()
print ("It took %d hours for training." % ((time_end-time_start)/3600))

# if lossgraph == 'true':
#     # plot loss graph
#     plt.figure(figsize=(10,5))
#     plt.title('Training Loss')
#     plt.plot(train_loss,label="loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.show()

