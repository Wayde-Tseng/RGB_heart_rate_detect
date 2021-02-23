import os
import numpy as np
import cv2

train_path = '/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_train_forehead.npy'
val_path = '/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_val_forehead.npy'
train = np.load(train_path)
for i, n in enumerate(train):

    for ii, nn in enumerate(n):
        img = cv2.imread(nn)
        img = cv2.resize(img, (150, 150))
        cv2.imwrite(nn, img)
        print(nn)
train = np.load(val_path)
for i, n in enumerate(train):

    for ii, nn in enumerate(n):
        img = cv2.imread(nn)
        img = cv2.resize(img, (150, 150))
        cv2.imwrite(nn, img)
        print(nn)