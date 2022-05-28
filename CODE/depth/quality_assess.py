from math import log10, sqrt
import cv2
import numpy as np
import os
import glob
from PIL import Image

path = './datasets/DodgeHeatWave_restored/'
path_pred = './results/dodgedepth8/model/'
list_psnr = []
list_ssim = []

def PSNR(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr

def SSIM(y_true, y_pred):
  u_true = np.mean(y_true)
  u_pred = np.mean(y_pred)
  var_true = np.var(y_true)
  var_pred = np.var(y_pred)
  std_true = np.sqrt(var_true)
  std_pred = np.sqrt(var_pred)
  c1 = np.square(0.01 * 7)
  c2 = np.square(0.03 * 7)
  ssim = (2 * u_true * u_pred + c1) * (2 * std_pred * std_true + c2)
  denom = (u_true ** 2 + u_pred ** 2 + c1) * (var_pred + var_true + c2)
  return ssim / denom

filesnames = glob.glob(os.path.join(path,'*.png'))
for i in range(len(filesnames)):
  curfile = filesnames[i]
  subname = curfile.split("\\")
  # print(subname[1])
  gt = Image.open(path + subname[1])
  pred = Image.open(path_pred + subname[1])
  gt = np.array(gt)
  pred = np.array(pred)
  psnrvalue = PSNR(gt,pred)
  ssimvalue = SSIM(gt,pred)
  list_psnr.append(psnrvalue)
  list_ssim.append(ssimvalue)
# print(list_psnr)
print('average psnr:',np.mean(list_psnr))
print('average ssim:',np.mean(list_ssim))

       
