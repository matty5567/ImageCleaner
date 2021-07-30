import datetime
import cv2
import numpy as np
import math

cap = cv2.VideoCapture('vids/people_walking.mp4')

frames = []
counter = 0

length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

while(cap.isOpened()):


    ret, frame = cap.read()

    if counter % math.floor(length/15) == 0:
        frames.append(frame)


    if ret == False:
        frames.pop()
        break

    counter +=1

cap.release()

cv2.destroyAllWindows()


print("number of frames", len(frames))

# median
np_frames = np.array(frames)


final_img = np.median(np_frames, axis=0)

#%%
cv2.imwrite(f'output/image{datetime.datetime.now()}.jpg', final_img)
cv2.imshow('Frame',final_img/255)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
