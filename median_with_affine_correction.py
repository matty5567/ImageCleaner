import datetime
import cv2
import numpy as np
import math
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(description='')
parser.add_argument('--name',required=True)

name = parser.parse_args().name

cap = cv2.VideoCapture(f'vids/{name}.mp4')

frames = []
counter = 0
median_frames_counter = 0

NUM_MEDIAN_FRAMES = 15

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

adjusted_frames = np.empty((NUM_MEDIAN_FRAMES+1, h, w, 3))

success, frame = cap.read()

if not success:
    Exception("Video not found")

else:
    first_frame = np.array(frame)
    first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    first_pts = cv2.goodFeaturesToTrack(first_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
    adjusted_frames[0, :, :, :] = first_frame


while(cap.isOpened()):
    

    success, frame = cap.read()

    if not success:
        break

    if counter % math.ceil(length/NUM_MEDIAN_FRAMES) == 0:
        median_frames_counter += 1
        curr_frame = np.array(frame)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(first_gray, curr_gray, first_pts, None)

        idx = np.where(status==1)[0]

        first_pts = first_pts[idx]
        curr_pts = curr_pts[idx]

        m = cv2.estimateAffine2D(first_pts, curr_pts)[0]

        invert = cv2.invertAffineTransform(m)

        adjusted_frames[median_frames_counter, :, :, :] = cv2.warpAffine(curr_frame, invert, (w, h))

    counter += 1

cap.release()

cv2.destroyAllWindows()


final_img = np.median(adjusted_frames, axis=0)


cv2.imwrite(f'output/image{datetime.datetime.now()}.jpg', final_img)
cv2.imshow('Frame',final_img/255)
cv2.waitKey(0)
cv2.destroyAllWindows()


