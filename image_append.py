import os
import cv2
from matplotlib.pyplot import axis
import numpy as np 


def img_info(path):
  img = cv2.imread(path)
  print(img.shape)
  (height,width,depth) = img.shape

  # print("image hight:",height)
  # print("image width:",width)
  # print("image depth:",depth)
  return height,width,depth

def y_append(path):
  dirs = os.listdir(path)
  output_image = np.zeros((384,159,3), np.uint8)
  y_t = np.zeros((384,3), np.uint8)
  indout = 0
  for item in dirs:
    if os.path.isfile(path+item):
      img = cv2.imread(path+item)
      column_y = img[:,100]
      output_image[:,indout,:] = column_y
      indout = indout + 1
  cv2.imshow('image',output_image)
  cv2.imwrite('vandistort.jpg', output_image)
  cv2.waitKey(0)  

def x_append(path):
  dirs = os.listdir(path)
  output_image = np.zeros((159,480,3), np.uint8)
  # x_t = np.zeros((134,3), np.uint8)
  indout = 0
  for item in dirs:
    if os.path.isfile(path+item):
      img = cv2.imread(path+item)
      row_x = img[250,:]
      output_image[indout,:,:] = row_x
      indout = indout + 1
  cv2.imshow('image',output_image)
  cv2.imwrite('vandistort_yt.jpg', output_image)
  cv2.waitKey(0)

if __name__ == '__main__':
  # path = './van_distorted/'
  path = './van_distorted/'
  # y_append(path)
  x_append(path)
    