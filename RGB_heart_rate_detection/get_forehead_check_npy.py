from ford_utils import load_npy, save_npy, printf, put_text, custom_sort
from glob import glob
import numpy as np
import cv2
import os
from argparse import ArgumentParser

#train_outliers = ['_2', '_3']
#valid_outliers = ['_1', '_3']

#train_outliers = ['data0827/sub4/150',
                  'data0828/sub2',
                  'data0910',
                  'data0912',
                  '_1']
#valid_outliers = ['data0827/sub4',
                  'data0828/sub2',
                  'data0910',
                  'data0912',
                  'data0827',
                  'data0828',
                  'sub1',
                  'sub3',
                  'sub5'
                  '_2',
                  '_3']
sample_time = 10
no_face = 0


def read_gt(gt_path):
    with open(gt_path, 'r') as f:
        lines = f.readlines()
    gt_second = list(map(int, lines))

    return gt_second


def mapping_gt_with_frame_num(gt_second, frame_num):
    secs = len(gt_second)
    gap = frame_num / secs
    intervals = list(map(int, np.arange(0, frame_num + 1, gap)))
    gt_frame = np.zeros((frame_num, 1))
    for i in range(secs):
        gt_frame[intervals[i]:intervals[i + 1]] = gt_second[i]
    gt_frame = list(map(int, gt_frame))
    return gt_frame


def sample_frame(frame_num, ori_video_fps, desired_video_fps):
    # frame_num -= 1
    sample_gap = ori_video_fps / desired_video_fps
    sampled_frame_idx = list(map(int, np.arange(0, frame_num, sample_gap)))
    return sampled_frame_idx


def exclude_outlier(path, outliers):
    if not os.path.isdir(path):
        return False

    for outlier in outliers:
        if outlier in path:
            return False

    return True


train_dirs = []
valid_dirs = []
fourcc = cv2.VideoWriter_fourcc(*'XVID')


sample_time = sample_time  # seconds

ori_video_fps = 1
desired_video_fps = 8
desired_signal_len = desired_video_fps * sample_time
save_npy_path = './train_data/train0714/testimg/'
save_npy_path = './save_npy/'

root_dir = './hr/data'
dir_dates = custom_sort(glob(root_dir + '/*'))
for dir_date in dir_dates:
    dir_subjects = custom_sort(glob(dir_date + '/*'))
    for dir_subject in dir_subjects:
        dir_speeds = custom_sort(glob(dir_subject + '/*'))
        for dir_speed in dir_speeds:
            if exclude_outlier(dir_speed, train_outliers):
                print('train dir : {}'.format(dir_speed))
                train_dirs.append(dir_speed)
            if exclude_outlier(dir_speed, valid_outliers):
                print('valid dir : {}'.format(dir_speed))
                valid_dirs.append(dir_speed)

print('train dir number : {}'.format(len(train_dirs)))
print('valid dir number : {}'.format(len(valid_dirs)))

datasets = {'train_dirs': train_dirs, 'valid_dirs': valid_dirs}

all_train_video_forehead = []
all_train_video_check = []
all_valid_video_forehead = []
all_Valid_video_check = []
all_train_imgs_forehead = []
all_train_imgs_check = []
all_train_gts = []
all_valid_imgs_forehead = []
all_valid_imgs_check = []
all_valid_gts = []

for key in datasets:
    dirs = datasets.get(key)
    for nth_dir, dir_path in enumerate(dirs):
        frame_num = len(glob(dir_path + '/*.png'))

        npy_path = dir_path + '/face_rects2.npy'
        face_rects = list(load_npy(npy_path))

        sampled_frame_idx = sample_frame(frame_num, ori_video_fps, desired_video_fps)

        gt_path = dir_path.replace('/data/', '/gt/') + '/gt_point.txt'
        gt_second = read_gt(gt_path)
        gt_frame = mapping_gt_with_frame_num(gt_second, frame_num)
        print(dir_path)
        forehead_path = []
        check_path = []
        gts = []
        no = 0
        forehead_video_path = dir_path + '/face/forehead/forehead_video.avi'
        check_video_path = dir_path + '/face/check/check_video.avi'
        write_video_forehead = cv2.VideoWriter(forehead_video_path, fourcc, 20.0, (140, 40))
        write_video_check = cv2.VideoWriter(check_video_path, fourcc, 20.0, (140, 40))

        for frame_idx, face_rect in enumerate(face_rects):
            imageBGR = cv2.imread(face_rect['image_path'])
            #print(face_rect['image_path'])
            print('{} : {} / {} : {} / {}'.format(dir_path, nth_dir + 1, len(dirs), frame_idx + 1, frame_num))
            canvas = imageBGR.copy()
            save_path = dir_path + '/face'
            save_forehead = save_path + '/forehead/'
            save_check = save_path + '/check/'
            if not face_rect['face_detected']:
                if no_face == 0:
                    print('[INFO] No face detected.')
                    g_value = None
                    no += 1
            else:
                face_point1 = (face_rect['face']['x1'], face_rect['face']['y1'])
                face_point2 = (face_rect['face']['x2'], face_rect['face']['y2'])
                crop_face = imageBGR[face_point1[1]:face_point2[1], face_point1[0]:face_point2[0]]
                crop_face = cv2.resize(crop_face, (140, 120), interpolation=cv2.INTER_LINEAR)
                canvas2 = crop_face.copy()
                forehead = crop_face[0:40, 0:140]
                # cv2.imshow('forehead', forehead)
                check = crop_face[60:100, 0:140]
                # cv2.imshow('check', check)
                cv2.rectangle(canvas2, (0, 0), (140, 40), (0, 255, 0), 2)
                cv2.rectangle(canvas2, (0, 60), (140, 100), (0, 0, 255), 2)
                # cv2.imshow('test', canvas2)
                #
                save_forehead = save_forehead + str(frame_idx + 1 - no) + '.png'
                save_check = save_check + str(frame_idx + 1 - no) + '.png'
                save_face = save_path + '/' + str(frame_idx + 1 - no) + '.png'
                cv2.imwrite(save_face, crop_face, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                cv2.imwrite(save_forehead, forehead, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                write_video_forehead.write(forehead)
                aaa = forehead.shape
                print(aaa)
                forehead_path.append(save_forehead)
                cv2.imwrite(save_check, check, [cv2.IMWRITE_PNG_COMPRESSION, 5])
                write_video_check.write(check)
                aaa = check.shape
                print(aaa)
                check_path.append(save_check)
                gts.append(gt_frame[frame_idx])
                cv2.waitKey(1)

        if key == 'train_dirs':
            all_train_video_forehead.append(forehead_video_path)
            all_train_video_check.append(check_video_path)
        elif key == 'valid_dirs':
            all_valid_video_forehead.append(forehead_video_path)
            all_Valid_video_check.append(check_video_path)


        for i in range(len(forehead_path)):
            imgss_forehead = []
            imgss_check = []
            imgss_forehead = forehead_path[i:i + desired_signal_len]
            imgss_check = check_path[i:i + desired_signal_len]
            if None in imgss_forehead or len(imgss_forehead) != desired_signal_len:
                continue

            print(imgss_forehead)
            if key == 'train_dirs':
                all_train_imgs_check.append(imgss_check)
                all_train_imgs_forehead.append(imgss_forehead)
                all_train_gts.append(gts[i])
            elif key == 'valid_dirs':
                all_valid_imgs_check.append(imgss_check)
                all_valid_imgs_forehead.append(imgss_forehead)
                all_valid_gts.append(gts[i])

tra_video_forehead = save_npy_path + 'all_train_video_forehead.npy'
tra_video_check = save_npy_path + 'all_train_video_check.npy'
val_video_forehead = save_npy_path + 'all_valid_video_forehead'
val_video_check = save_npy_path + 'all_valid_video_check.npy'
tra_forehead_dir = save_npy_path + 'all_train_forehead.npy'
tra_check_dir = save_npy_path + 'all_train_check.npy'
val_forehead_dir = save_npy_path + 'all_valid_forehead.npy'
val_check_dir = save_npy_path + 'all_valid_check.npy'
tra_gt_dir = save_npy_path + 'all_train_gts.npy'
val_gt_dir = save_npy_path + 'all_valid_gts.npy'



save_npy(tra_video_forehead, all_train_video_forehead)
save_npy(tra_video_check, all_train_video_check)
save_npy(val_video_forehead, all_valid_video_forehead)
save_npy(val_video_check, all_Valid_video_check)
save_npy(tra_forehead_dir, all_train_imgs_forehead)
save_npy(tra_check_dir, all_train_imgs_check)
save_npy(tra_gt_dir, all_train_gts)
save_npy(val_forehead_dir, all_valid_imgs_forehead)
save_npy(val_check_dir, all_valid_imgs_check)
save_npy(val_gt_dir, all_valid_gts)
