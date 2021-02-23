import numpy as np
import os
import cv2
import pandas as pd

dataset_dlr="/home/wayde/Desktop/DATASET_2/"

def sort_subject_dlr(subject_list):
    subject=[]
    count=1

    for i in subject_list:
        if (len(i)==8):
            i=i.rstrip(str(count))
            i=i+'0'+str(count)

            count+=1
            if(count==2):
                count+=1
            if(count==6):
                count+=1
            if(count==7):
                count+=1
        subject.append(i)

    subject.sort(key=lambda x: int(x[7:9]))
    count=1
    for i in range(0,6):
        num=str(count)
        subject[i]=subject[i].rstrip('0'+num)
        subject[i]=subject[i]+num
        count += 1
        if (count == 2):
            count += 1
        if (count == 6):
            count += 1
        if (count == 7):
            count += 1
    return subject




subject_list=os.listdir(dataset_dlr)
subject_list = sorted(subject_list)
#subject_list=sort_subject_dlr(subject_list)
#print(subject_list)

train_forehead_path=[]
#train_cheek_path=[]

val_forehead_path=[]
#val_cheek_path=[]
print(len(subject_list))
count=0
for i in subject_list:
    subject_path=os.path.join(dataset_dlr,i)
    forehead_dlr=os.path.join(subject_path,"face")
    #cheek_dlr=os.path.join(subject_path,"cheek")
    print(forehead_dlr)
    forehead_list=os.listdir(forehead_dlr)
    #print(forehead_list)
    #print(forehead_list)
    #cheek_list=os.listdir(cheek_dlr)

    if(count<35):
        for j in range(0,len(forehead_list)-80):
            eighty_stack=[]
            for k in range(j,j+80):
                k=str(k)+'.jpg'
                k=os.path.join(forehead_dlr,k)
                eighty_stack.append(k)
            train_forehead_path.append(eighty_stack)

        # for j in range(0,len(cheek_list)-80):
        #     eighty_stack=[]
        #     for k in range(j,j+80):
        #         k=str(k)+'.jpg'
        #         k=os.path.join(cheek_dlr,k)
        #         eighty_stack.append(k)
        #     train_cheek_path.append(eighty_stack)
    else:
        for j in range(0,len(forehead_list)-80):
            eighty_stack = []
            for k in range(j, j + 80):
                k = str(k) + '.jpg'
                k = os.path.join(forehead_dlr, k)
                eighty_stack.append(k)
            val_forehead_path.append(eighty_stack)

        # for j in range(0,len(cheek_list)):
        #     eighty_stack = []
        #     for k in range(j,j+80):
        #         k =str(k) + '.jpg'
        #         k=os.path.join(cheek_dlr,k)
        #         eighty_stack.append(k)
        #     val_cheek_path.append(eighty_stack)

    count=count+1
    #print(count)



train_forehead_path=np.array(train_forehead_path)
np.save('/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_train_forehead.npy',train_forehead_path)


#train_cheek_path=np.array(train_cheek_path)
#np.save( 'C:\\Users\lab70636\Desktop\RGB\program\save_npy\\all_train_cheek.npy',train_cheek_path)


val_forehead_path=np.array(val_forehead_path)
np.save('/home/wayde/Desktop/RGB_heart_rate_detect/save_npy/all_val_forehead.npy',val_forehead_path )


#val_cheek_path=np.array(val_cheek_path)
#np.save( 'C:\\Users\lab70636\Desktop\RGB\program\save_npy\\all_val_cheek.npy',val_cheek_path)





