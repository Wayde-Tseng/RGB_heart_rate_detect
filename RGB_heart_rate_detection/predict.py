
import numpy as np
import torch
from siamese_network import SIAMESE
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


all_error = []
all_error_2 = []
sample_length = 80
write_video = 1
show_hr = 1

model = SIAMESE()
model.load_state_dict(torch.load('./model/Siamese_noShareWeights -24 -11.840457027668016.pkl'))
model.cuda()
model.eval()
# print(model)

#total_path_face = np.load('./npy_path/test/face.npy')
total_path_forehead = np.load('./save_npy/all_val_forehead.npy')
#total_path_check = np.load('./npy_path/test/check.npy')
total_gt = np.load('./save_npy/all_valid_gts.npy')
#print(total_path_forehead)




result_path = './result/'
test_fold = './test_video/'


gt_array = []
ans_array = []
all_e = 0
all_e2 = 0


for nth_subject in range(len(total_path_forehead)):
    forehead_paths = total_path_forehead[nth_subject]
    gt = total_gt[nth_subject]

    error = []
    forehead_imgs = []
    #check_imgs = []

    transform = transforms.Compose([
        transforms.ToTensor()])
    # ---------------------------
    #change video
    # ---------------------------
    if write_video == 1:
        if '/0.jpg' in total_path_forehead[nth_subject][0]:
            if nth_subject != 0:
                print(predict_array)
                plt.subplot(211)
                plt.plot(predict_array, 'r')
                plt.plot(gt_array, 'g')
                plt.legend((nth_subject,'predict', 'ground truth'), shadow=True, loc=(0.01, 0.01))
                plt.subplot(212)
                plt.plot(error_array, 'r')
                plt.plot(ans_array, 'g')
                plt.legend(('error', 'ground truth'), shadow=True, loc=(0.01, 0.01))
                plt.pause(0.01)
                plt.savefig(result_path + fold_id + '_Siamese_result.jpg')
                plt.close()
            predict_array = []
            ans_array = []
            gt_array = []
            error_array = []
            pp = total_path_forehead[nth_subject][0]
            print(pp)
            #if gt>=100:
            #    fold_id = pp[30:-37]
            #else:
            #    fold_id = pp[15:-36]
            #if nth_subject
            fold_id = pp[30:-11]
            print(fold_id)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            video_path = result_path + fold_id + '_result.avi'
            videowriter = cv2.VideoWriter(video_path, fourcc, 30.0, (480, 640))
    # write_video_forehead = cv2.VideoWriter(forehead_video_path, fourcc, 20.0, (140, 40))
    # ---------------------------
    gt_array.append(gt)
    ans_array.append(gt)
    for nth_img in range(1, len(forehead_paths)):
        forehead_path = total_path_forehead[nth_subject][nth_img]
        #check_path = total_path_check[nth_subject][nth_img]
        #face_path = total_path_face[nth_subject][nth_img]


        imgBGR = cv2.imread(forehead_path)
        forehead_img = Image.open(forehead_path)
        #check_img = Image.open(check_path)
        forehead_img = transform(forehead_img)
        forehead_img = forehead_img.cuda()
        forehead_img = torch.unsqueeze(forehead_img, 1)
        #check_img = transform(check_img)
        #check_img = check_img.cuda()
        #check_img = torch.unsqueeze(check_img, 1)

        if nth_img == 1:
            video_forehead = forehead_img
            #video_check = check_img
        else:
            video_forehead = torch.cat([video_forehead, forehead_img], 1)
            #video_check = torch.cat([video_check, check_img], 1)

        # ---------------------------
        # make every 80 frame together
        # ---------------------------
    video_forehead = torch.unsqueeze(video_forehead, 0)
    #video_check = torch.unsqueeze(video_check, 0)
    aa = model(video_forehead)
    aa = aa.squeeze(-1)
    aa = aa.squeeze(-1)
    aa = aa.squeeze(-1)
    aa = aa.squeeze(-1)
    ans = int(aa)

    # ---------------------------
    # check the ans
    # ---------------------------
    error = abs(ans - gt)
    predict_array.append(ans)
    error_array.append(error)
    # ---------------------------
    # show the testing result
    # ---------------------------
    # ---------------------------
    # show the testing result
    # ---------------------------
    if show_hr == 1:
        imgBGR = cv2.putText(imgBGR, 'predict : ' + str(np.round(ans, 3)), (5, 15), cv2.FONT_HERSHEY_PLAIN,
                             0.7,
                             (0, 255, 255))
        imgBGR = cv2.putText(imgBGR, 'GT : ' + str(int(np.round(gt))), (5, 25), cv2.FONT_HERSHEY_PLAIN, 0.7,
                             (0, 255, 255))
        if error > 3:
            color = (0, 0, 255)
            all_e = all_e + 1
            if error > 5:
                all_e2 = all_e2 + 1
                color = (0, 0, 255)
        else:
            color = color = (0, 255, 255)
        imgBGR = cv2.putText(imgBGR, 'error : ' + str(np.round(error, 3)), (5, 35), cv2.FONT_HERSHEY_PLAIN,
                             0.7, color)
    # ---------------------------
    # write video
    # ---------------------------
    if write_video == 1:
        imgBGR = cv2.resize(imgBGR, (480, 640))
        video = videowriter.write(imgBGR)
    # ---------------------------

    cv2.imshow('win', imgBGR)
    cv2.waitKey(1)



    all_error.append(np.mean(error_array))
    all_error_2.append((np.mean(error_array) / np.mean(ans_array) * 100))

print(np.mean(all_error))
print(np.mean(all_error_2))
print(all_e / len(gt_array))
print(all_e2 / len(gt_array))
