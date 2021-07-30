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

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


while(cap.isOpened()):


    success, frame = cap.read()

    if not success:
        break

    if counter % math.floor(length/15) == 0:
        frames.append(frame)

    counter +=1


cap.release()

cv2.destroyAllWindows()

num_frames = len(frames)

print("number of frames", num_frames)

# median
np_frames = np.array(frames)

adjusted_frames = np.empty_like(np_frames)


transforms = np.zeros((num_frames, 3), np.float32)

first_gray = cv2.cvtColor(np_frames[0, :, :, :], cv2.COLOR_BGR2GRAY)

for i in tqdm(range(num_frames)):
    first_pts = cv2.goodFeaturesToTrack(first_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)

    curr_gray = cv2.cvtColor(np_frames[i, :, :, :], cv2.COLOR_BGR2GRAY)

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(first_gray, curr_gray, first_pts, None)

    assert first_pts.shape == curr_pts.shape

    idx = np.where(status==1)[0]
    
    first_pts = first_pts[idx]
    curr_pts = curr_pts[idx]

    m = cv2.estimateAffine2D(first_pts, curr_pts)[0]

    dx = m[0, 2]
    dy = m[1, 2]

    da = np.arctan2(m[1, 0], m[0, 0])

    transforms[i] = [dx, dy, da]

    invert = cv2.invertAffineTransform(m)

    adjusted_frames[i, :, :, :] = cv2.warpAffine(np_frames[i, :, :, :], invert, (w, h))

final_img = np.median(adjusted_frames, axis=0)

cv2.imwrite(f'output/image{datetime.datetime.now()}.jpg', final_img)
cv2.imshow('Frame',final_img/255)
cv2.waitKey(0)
cv2.destroyAllWindows()


