
import os, sys
import cv2
import time

def video_to_frames(input_loc, output_loc):
    """Extract frames from input video file
    and save them as separate frames in an output directory.
    Args:
        input_loc: Input video file.
        output_loc: Output directory to save the frames.
    Returns:
        None
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


# def videoframe(path):
#     if not os.path.exists('frames'):
#         os.mkdir('frames')
#     vidcap = cv2.VideoCapture(path)
#     success,image = vidcap.read()
#     count = 0
#     while success:
#         cv2.imwrite("./frames/frame%d.png" % count, image)     
# # save frame as png file      
#         success,image = vidcap.read()
#         print('Read a new frame: ', success)
#         count += 1

# path = './datasets/output_distorted.mp4/'
# videoframe(path)


input_loc = './datasets/output_distorted.mp4/'
output_loc = './frames/'
video_to_frames(input_loc, output_loc)

