                                                              
import numpy as np
import cv2
import os, sys
import glob
import time

from os.path import isfile, join

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
            print(filename)
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


path = './results/model/'
pathOut = 'video.avi'
fps = 25.0

frames_to_video(path,pathOut,fps)

