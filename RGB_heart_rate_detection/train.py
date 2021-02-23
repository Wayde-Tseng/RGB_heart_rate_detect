import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from video_dataloader_faceonly import customDataset, transform
from torch.utils.data import DataLoader
import time
from siamese_network_no_share_new_attention_face import SIAMESE_no_share, pearson_correlation
#from siamese_network_no_share import SIAMESE_no_share
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter



EPOCH = 200
BATCH_SIZE = 3
LR = 0.0001
# give forehead check and PPG
npy_path = './save_npy/'
train_forehead_path = npy_path + 'all_train_forehead.npy'
#train_check_path = npy_path + 'all_train_check.npy'
train_gts = npy_path + 'all_train_gts.npy'
valid_forehead_path = npy_path + 'all_valid_forehead.npy'
#valid_check_path = npy_path + 'all_valid_check.npy'
valid_gts = npy_path + 'all_valid_gts.npy'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_loader = customDataset(
    root_forehead='./save_npy/all_train_forehead.npy', #root_check='./save_npy/all_train_check.npy',
    root_gts='./save_npy/all_train_gts.npy', split='train'
    , transform=transform)
val_loader = customDataset(
    root_forehead='./save_npy/all_val_forehead.npy', #root_check='./save_npy/all_valid_check.npy',
    root_gts='./save_npy/all_valid_gts.npy', split='valid'
    , transform=transform)

train_dataloader = DataLoader(train_loader, batch_size=24, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_loader, batch_size=24, shuffle=True, num_workers=0)

siamese = SIAMESE_no_share().cuda()
#siamese = SIAMESE()
#siamese = torch.load('./Siamese -35 -best').cuda()
siamese.train()

print(siamese)
loss_function = nn.L1Loss()



optimizer = torch.optim.Adam(siamese.parameters(), lr=LR)
error_list = []
writer = SummaryWriter()
for epoch in range(EPOCH):
    step = 0
    for forehead, gt in train_dataloader:
        step = step + 1
        output = siamese(forehead)
        output = output.squeeze(-1)
        output = output.squeeze(-1)
        output = output.squeeze(-1)
        output = output.squeeze(-1)
        train_loss = loss_function(output, gt.cuda())
        writer.add_scalar('training_Loss', train_loss, step)
        #train_loss = pearson_correlation(output, gt.cuda().float())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        print('Epoch:{} | num: {} / {} | L1Loss: {} '.format(epoch, step, len(train_dataloader), train_loss))
    val_step = 0
    sum_val_loss = 0
    siamese.eval()
    with torch.no_grad():
        for val_forehead, gt in val_dataloader:
            val_step = val_step + 1
            test_output = siamese(val_forehead)
            test_output = test_output.squeeze(-1)
            test_output = test_output.squeeze(-1)
            test_output = test_output.squeeze(-1)
            test_output = test_output.squeeze(-1)
            val_loss = loss_function(test_output, gt.cuda())
            #val_loss = pearson_correlation(test_output, gt.cuda().float())
            sum_val_loss = sum_val_loss + val_loss
    mean_val_loss = sum_val_loss/val_step
    error_list.append(mean_val_loss)
    # plt.close()
    # plt.plot(error_list)
    # plt.show()
    writer.add_scalar('val_Loss', mean_val_loss, step)
    siamese.train()
    torch.save(siamese.state_dict(), './model/Siamese_noShareWeights -{} -{}.pkl'.format(epoch, mean_val_loss.item()))
plt.savefig("/home/wayde/Desktop/RGB_heart_rate_detect/loss.jpg")
plt.close()