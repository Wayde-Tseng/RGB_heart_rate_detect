import os
import numpy as np
import matplotlib.pyplot as plt

path = './hr/data/'
paths = os.listdir(path)
paths = sorted(paths)
paths = paths[:-2]
all_gt = np.zeros(210)
#print(all_gt)
#print(paths)
for data_path in paths:
    date_path = path + data_path + '/'
    gt_fold = os.listdir(date_path)
    gt_fold = sorted(gt_fold)
    #print(gt_fold)
    for fold_id in gt_fold:
        id_path = date_path + fold_id + '/'
        gt_paths = os.listdir(id_path)

        for gt_path in gt_paths:
            final_path = './hr/gt/' + data_path + '/' + fold_id + '/' + gt_path + '/gt_cal.txt'
            with open(final_path, 'r') as f:
                gt_second = f.readlines()
                #print(gt_second)
                gtss = list(map(int, gt_second))
                #gtss = gtss[80:]
                #print(gtss)
                for gt in gtss:
                    #print(gt)
                    all_gt[gt] = all_gt[gt] + 1

            #print(final_path)
plt.plot(all_gt)
plt.show()