from ford_utils import custom_sort
from glob import glob
import os

def exclude_outlier(path, outliers):
    if not os.path.isdir(path):
        return False

    for outlier in outliers:
        if outlier in path:
            return False

    return True

def not_exclude_outlier(path, outliers):
    if not os.path.isdir(path):
        return False

    for outlier in outliers:
        if outlier  in path:
            return True

    return False


def get_path():
    total_path = []

    root_dir = './hr/data'
    dir_dates = custom_sort(glob(root_dir + '/*'))
    for dir_date in dir_dates:
        dir_subjects = custom_sort(glob(dir_date + '/*'))
        for dir_subject in dir_subjects:
            dir_speeds = custom_sort(glob(dir_subject + '/*'))
            for dir_speed in dir_speeds:
                total_path.append(dir_speed)

    return total_path


def get_path_test():
    total_path = []
    test_outliers = ['_3', '_2', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3'
                     , '25_', '50_', '75_', '100_', '150_']
#    test_outliers = ['_1', '_2', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3'
#                     , '0_', '50_', '75_', '100_', '150_']
#    test_outliers = ['_1', '_2', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3'
#                     , '25_', '75_', '0_', '100_', '150_']
#    not_test_outliers = ['50_3']
#    not_test_outliers = ['100_3']
#    test_outliers = ['_1', '_2', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3'
#                     , '25_', '50_', '100_', '0_', '150_']
#    test_outliers = ['_1', '_2', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3'
#                     , '0_', '50_', '75_', '100_', '25_']
    root_dir = './hr/data'
    dir_dates = custom_sort(glob(root_dir + '/*'))
    for dir_date in dir_dates:
        dir_subjects = custom_sort(glob(dir_date + '/*'))
        for dir_subject in dir_subjects:
            dir_speeds = custom_sort(glob(dir_subject + '/*'))
            for dir_speed in dir_speeds:
                #if exclude_outlier(dir_speed, test_outliers):
                #    total_path.append(dir_speed)
                if not_exclude_outlier(dir_speed, test_outliers):
                   total_path.append(dir_speed)

    return total_path


def get_path_test_n():
    total_path = []
    test_outliers = ['_1', '_2', '150', '100', 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3']

    root_dir = './hr/data'
    dir_dates = custom_sort(glob(root_dir + '/*'))
    for dir_date in dir_dates:
        dir_subjects = custom_sort(glob(dir_date + '/*'))
        for dir_subject in dir_subjects:
            dir_speeds = custom_sort(glob(dir_subject + '/*'))
            for dir_speed in dir_speeds:
                if exclude_outlier(dir_speed, test_outliers):
                    total_path.append(dir_speed)

    return total_path


def get_path_test_new():
    total_path = []
    test_outliers = ['data0827/sub4',
                  'data0828/sub2',
                  'data0910',
                  'data0912',
                  'data0827',
                  'data0828',
                  'sub2',
                  'sub4',
                  '_2',
                  '_3'
                     , 'data0828/sub4/100_3', 'data0828/sub5/100_3', 'data0829/sub1/75_3', 'data0829/sub1/100_3']

    root_dir = './hr/data'
    dir_dates = custom_sort(glob(root_dir + '/*'))
    for dir_date in dir_dates:
        dir_subjects = custom_sort(glob(dir_date + '/*'))
        for dir_subject in dir_subjects:
            dir_speeds = custom_sort(glob(dir_subject + '/*'))
            for dir_speed in dir_speeds:
                if exclude_outlier(dir_speed, test_outliers):
                    total_path.append(dir_speed)

    return total_path

get_path_test_new()
