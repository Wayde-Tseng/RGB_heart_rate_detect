import os
import cv2
import numpy as np
from SSD.ssd import SSD300
from SSD.SSD_input import process_head_region
import pickle
import matplotlib.pyplot as plt

input_shape = (300, 300, 3)
SSD_model = SSD300(input_shape, num_classes=2)
SSD_model.load_weights('hr_face_model.hdf5')

total_path = []


total_path.append('./hr/data/data0827/sub1/0_2')
# total_path.append('./hr/data/data0827/sub1/25_2')
# total_path.append('./hr/data/data0827/sub1/50_2')
# total_path.append('./hr/data/data0827/sub1/75_2')
# total_path.append('./hr/data/data0827/sub1/100_2')


for nth_subject in range(len(total_path)):

    folder_path = total_path[nth_subject]

    frame_cnt = 0
    for my_file in os.listdir(folder_path):  # read all file in flodr
        if my_file.endswith('.png'):
            frame_cnt += 1  # count how may .png

    f_gt = open('./hr/gt/' + folder_path[10:] + '/gt_cal.txt', 'r')
    lines = f_gt.readlines()
    f_gt.close()

    gt_array = []
    for line in lines:
        line = int(line.split('\n')[0])
        gt_array.append(line)

    skip_cnt = 0
    ori_imgs = []
    imcs = []
    gts = []

    # for i in range(1, frame_cnt + 1):
    for i in range(1, len(gt_array) + 1):
        # for i in range(1, 50):

        print(folder_path + ' : ' + str(i) + ' / ' + str(frame_cnt))
        print(str(nth_subject) + ' / ' + str(len(total_path)))

        img = cv2.imread(folder_path + '/' + str(i) + '.png')
        img_re = cv2.resize(img, (300, 300))
        xmin, ymin, xmax, ymax, score, label = process_head_region(SSD_model, img_re)

        if not xmin:
            skip_cnt += 1
            print('skip')
            continue

        img_h, img_w, _ = img.shape
        xmin = int(xmin[0] / 300. * img_w)
        xmax = int(xmax[0] / 300. * img_w)
        ymin = int(ymin[0] / 300. * img_h)
        ymax = int(ymax[0] / 300. * img_h)

        canvas = img.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canvas = cv2.rectangle(canvas, (xmin, ymin), (xmax, ymax), (0, 255, 255))
        imc = gray[ymin:ymax, xmin:xmax]

        ori_imgs.append(imc)

        cv2.imshow('canvas', canvas)
        cv2.imshow('imc', imc)
        cv2.waitKey(1)

        imc = cv2.resize(imc, (32, 32))

        cv2.imshow('imc_re', imc)
        cv2.waitKey(1)

        imc = imc.astype('uint8')
        imcs.append(imc)
        gts.append(gt_array[i - 1])

    diff_imgs = []
    diff_gts = []

    for i in range(1, len(imcs)):
        current_frame = imcs[i]
        current_gt = gts[i]
        previous_frame = imcs[i - 1]
        diff_img = current_frame - previous_frame

        # cv2.imshow('current_frame', current_frame)
        # cv2.imshow('previous_frame', previous_frame)
        # cv2.imshow('diff_img', diff_img)
        # cv2.waitKey(0)

        diff_img = diff_img[np.newaxis, :]
        diff_gts.append(current_gt)

        if len(diff_imgs) == 0:
            diff_imgs = diff_img
        else:
            diff_imgs = np.concatenate((diff_imgs, diff_img), axis=0)

        print(diff_imgs.shape)
        print(diff_imgs, diff_gts, ori_imgs)

    with open(folder_path + '/diff_data.pkl', 'wb') as data_file:
        data = {'images': diff_imgs, 'labels': diff_gts, 'ori_imgs': ori_imgs}
        print('dump')
        pickle.dump(data, data_file, protocol=pickle.HIGHEST_PROTOCOL)
