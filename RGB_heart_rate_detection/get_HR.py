import os
import numpy as np
import pandas as pd


ground_truth=pd.read_csv("groundtruth.csv")

size=ground_truth["Each_vid_frame_size"][0:42]
HR=ground_truth['HR']
print (ground_truth)

count=0
train_HR=[]
frame=0
val_HR=[]

for i in size:
    i=int(i)
    if(count<35):
        frame = frame + 80
        for j in range(80,i+1):

            train_HR.append(HR[frame])

            frame+=1
    else:
        for j in range(80, i+1):
            val_HR.append(HR[frame])

            frame += 1

    count+=1
    #print(count)


train_HR=np.array(train_HR)
np.save("/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_train_gts.npy",train_HR)

val_HR=np.array(val_HR)
np.save("/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_valid_gts.npy",val_HR)
