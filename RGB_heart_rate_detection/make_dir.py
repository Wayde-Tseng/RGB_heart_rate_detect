import os
from ford_utils import custom_sort
from glob import glob


root_dir = './hr/data'
dir_dates = custom_sort(glob(root_dir + '/*'))
for dir_date in dir_dates:
    dir_subjects = custom_sort(glob(dir_date + '/*'))
    for dir_subject in dir_subjects:
        dir_speeds = custom_sort(glob(dir_subject + '/*'))
        for dir_speed in dir_speeds:
            dir_speed = dir_speed + '/face/'
            print(dir_speed)
            os.makedirs(dir_speed + 'check')
            os.makedirs(dir_speed + 'forehead')