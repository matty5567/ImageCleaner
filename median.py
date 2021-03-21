
import cv2
import numpy as np


cap = cv2.VideoCapture('test_vid.mp4')

frames = []
counter = 0

while(cap.isOpened()):


    ret, frame = cap.read()

    if counter % 30 == 0:
        frames.append(frame)


    if ret == False:
        frames.pop()
        break

        counter +=1

cap.release()

cv2.destroyAllWindows()


# median
np_frames = np.array(frames)
final_img = np.median(np_frames, axis=0)

#%%

cv2.imshow('Frame',final_img/255)
cv2.waitKey(0)
cv2.destroyAllWindows()


# %%
