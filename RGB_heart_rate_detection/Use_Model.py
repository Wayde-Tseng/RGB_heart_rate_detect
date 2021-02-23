from ford_utils import load_npy, save_npy, printf, put_text, custom_sort
from glob import glob
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision
import time
from get_path_2 import get_path_test, get_path_test_new
from siamese_network import SIAMESE
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# from SSD.ssd import SSD300
# from SSD.SSD_input import process_head_region

all_error = []
all_error_2 = []
sample_l = 80
# input_shape = (300, 300, 3)
# SSD_model = SSD300(input_shape, num_classes=2)
# SSD_model.load_weights('./hr_face_model.hdf5')
# python 3.9 no tensorflow

model = SIAMESE()
#model = torch.load('Siamese -35 -best').cuda()
model.load_state_dict(torch.load('./model/Siamese_noShareWeights -25 -5.403666019439697.pkl'))
model.cuda()
model.eval()
print(model)

total_path = []
total_path = get_path_test_new()
all_f = 0
all_e = 0
all_e2 = 0
for nth_subject in range(len(total_path)):
    folder_path = total_path[nth_subject]
    sample_length = sample_l
    frame_cnt = 0
    for my_file in os.listdir(folder_path):
        if my_file.endswith('.png'):
            frame_cnt = frame_cnt + 1

    f_gt = open('./hr/gt' + folder_path[9:] + '/gt_cal.txt', 'r')
    lines = f_gt.readlines()
    f_gt.close()

    gt_array = []

    for line in lines:
        line = line.split('\n')
        line = int(line[0])
        gt_array.append(line)

    max_er = 0
    skip_cnt = 0
    img = []
    error_array = []
    predict_array = []
    predict_img = []
    ans_array = []
    ori_sig_array = []
    fill_front = 0
    nth_frame = 0
    write_video = 1
    show_hr = 1
    transform = transforms.Compose([
        transforms.ToTensor()])

    result_path = './result/' + folder_path[19:]
    # ---------------------------
    if write_video == 1:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video_path = result_path + '_result.avi'
        videowriter = cv2.VideoWriter(video_path, fourcc, 20.0, (480, 640))
        # write_video_forehead = cv2.VideoWriter(forehead_video_path, fourcc, 20.0, (140, 40))
    # ---------------------------
    # read forehead and check region
    # ---------------------------
    forehead = []
    check = []
    forehead_path = folder_path + '/face/forehead/'
    #check_path = folder_path + '/face/check/'
    for i in range(1, frame_cnt + 1):
        all_f = all_f + 1
        print(folder_path + ' : ' + str(i) + '/' + str(frame_cnt))  # print now / all and folder path
        foreheadiii = forehead_path + str(i) + '.png'
        #checkiii = check_path + str(i) + '.png'
        imgBGR = cv2.imread(folder_path + '/' + str(i) + '.png')
        forehead_img = Image.open(foreheadiii)
        #check_img = Image.open(checkiii)
        forehead_img = transform(forehead_img)
        forehead_img = forehead_img.cuda()
        forehead_img = torch.unsqueeze(forehead_img, 1)
        #check_img = transform(check_img)
        #check_img = check_img.cuda()
        #check_img = torch.unsqueeze(check_img, 1)

        if i == 1:
            video_forehead = forehead_img
            #video_check = check_img
        else:
            video_forehead = torch.cat([video_forehead, forehead_img], 1)
            #video_check = torch.cat([video_check, check_img], 1)

        # ---------------------------

        # make every 80 frame together
        # ---------------------------
        if i >= sample_length and i % 1 == 0:
            if i ==80:
                predict_img_forehead = video_forehead[i - sample_length:i]
                #predict_img_check = video_check[i - sample_length:i]
            if i > 80:
                split_size = [i - sample_length, 80]
                predict_img_forehead = video_forehead.split(split_size, 1)
                predict_img_forehead = torch.tensor(predict_img_forehead[1])
                predict_img_forehead = predict_img_forehead.cuda()
                #predict_img_check = video_check.split(split_size, 1)
                #predict_img_check = torch.tensor(predict_img_check[1])
                #predict_img_check = predict_img_check.cuda()

            #predict_img_check = torch.unsqueeze(predict_img_check, 0)
            predict_img_forehead = torch.unsqueeze(predict_img_forehead, 0)
            # ---------------------------
            # predict the ans
            # ---------------------------

            aa = model(predict_img_forehead, predict_img_check)
            aa = aa.squeeze(-1)
            aa = aa.squeeze(-1)
            aa = aa.squeeze(-1)
            aa = aa.squeeze(-1)
            ans = int(aa)
            print(ans)
            # ---------------------------
            # check the ans
            # ---------------------------
            gt = gt_array[i - sample_length]
            error = abs(ans - gt)
            ans_array.append(gt)
            predict_array.append(ans)
            error_array.append(error)
            # ---------------------------
            # show the testing result
            # ---------------------------
            if show_hr == 1:
                imgBGR = cv2.putText(imgBGR, 'predict : ' + str(np.round(ans, 3)), (10, 25), cv2.FONT_HERSHEY_PLAIN,
                                     1.5,
                                     (0, 255, 255))
                imgBGR = cv2.putText(imgBGR, 'GT : ' + str(int(np.round(gt))), (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.5,
                                     (0, 255, 255))
                if error > 3:
                    color = (0, 0, 255)
                    all_e = all_e + 1
                    if error > 5:
                        all_e2 = all_e2 + 1
                        color = (0, 0, 255)
                else:
                    color = color = (0, 255, 255)
                imgBGR = cv2.putText(imgBGR, 'error : ' + str(np.round(error, 3)), (10, 75), cv2.FONT_HERSHEY_PLAIN,
                                     1.5, color)
            # ---------------------------
            # write video
            # ---------------------------
            if write_video == 1:
                imgBGR = cv2.resize(imgBGR, (480, 640))
                video = videowriter.write(imgBGR)
            # ---------------------------


            cv2.imshow('win', imgBGR)
            cv2.waitKey(1)

    # plot error img

    plt.subplot(211)
    plt.plot(predict_array, 'r')
    plt.plot(ans_array, 'g')
    plt.legend(('predict', 'ground truth'), shadow=True, loc=(0.01, 0.01))
    plt.subplot(212)
    plt.plot(error_array, 'r')
    plt.plot(ans_array, 'g')
    plt.legend(('error', 'ground truth'), shadow=True, loc=(0.01, 0.01))
    plt.pause(0.01)
    plt.savefig(result_path + '_Siamese_result.jpg')
    plt.close()


    all_error.append(np.mean(error_array))
    all_error_2.append((np.mean(error_array) / np.mean(ans_array) * 100))

print(total_path)
print(np.mean(all_error))
print(np.mean(all_error_2))
print(all_e /all_f)
print(all_e2 / all_f)
