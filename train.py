
# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;

import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from data import Dataset_Pro
from model import CML, loss_with_l2_regularization
import numpy as np
from torch.utils.tensorboard import SummaryWriter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
###################################################################
# ------------------- Pre-Define Part----------------------
###################################################################
# ============= 1) Pre-Define =================== #
SEED = 1
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
# cudnn.benchmark = True  ###自动寻找最优算法
cudnn.deterministic = True
cudnn.benchmark = False

# ============= 2) HYPER PARAMS(Pre-Defined) ==========#
lr = 0.0015
epochs = 1000
ckpt = 50
batch_size = 32
model_path = "Weights/wv3_newmulti/.pth"
gamma = 3/4
# ============= 3) Load Model + Loss + Optimizer + Learn_rate_update ==========#
model = CML().cuda()
if os.path.isfile(model_path):
    model.load_state_dict(torch.load(model_path))   ## Load the pretrained Encoder
    print('FusionNet is Successfully Loaded from %s' % (model_path))


criterion = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function

criterion_regular = loss_with_l2_regularization().cuda()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-8)   ## optimizer 1: Adam
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=200, gamma=gamma)

writer = SummaryWriter('./train_logs/cml')
def save_checkpoint(model, epoch):  # save model function
    model_out_path = 'Weights' + '/wv3_cml/' + "{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)

###################################################################
# ------------------- Main Train (Run second)----------------------
###################################################################
def train(training_data_loader, validate_data_loader,start_epoch=0):
    print('Start training...')
    model.train()

    for epoch in range(start_epoch, epochs, 1):
        flag = (epoch == (epochs-1)) or epoch == 0
        epoch += 1
        epoch_train_loss, epoch_val_loss = [], []

        # ============Epoch Train=============== #

        for iteration, batch in enumerate(training_data_loader, 1):
            # gt Nx8x64x64
            # ms Nx8x16x16
            # pan Nx1x64x64
            gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            optimizer.zero_grad()  # fixed
            sr = model(ms, pan)  # call model
            loss = criterion(sr, gt)  # compute loss
            new_loss = criterion_regular(loss, model, flag=flag)#0.0014
            epoch_train_loss.append(loss.item())  # save all losses into a vector for one epoch
            new_loss.backward()  # fixed
            optimizer.step()  # fixed

        lr_scheduler.step()  # if update_lr, activate here!

        t_loss = np.nanmean(np.array(epoch_train_loss))  # compute the mean value of all losses, as one epoch loss
        writer.add_scalar('mse_loss/t_loss', t_loss, epoch)  # write to tensorboard to check
        print('Epoch: {}/{} training loss: {:.7f}'.format(epochs, epoch, t_loss))  # print loss for each epoch

        if epoch % ckpt == 0:  # if each ckpt epochs, then start to save model
            save_checkpoint(model, epoch)

         # ============Epoch Validate=============== #
        model.eval()
        with torch.no_grad():  # fixed
             for iteration, batch in enumerate(validate_data_loader, 1):
                 gt, ms, pan = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
                 sr = model(ms, pan)  # call model
                 loss = criterion(sr, gt)
                 epoch_val_loss.append(loss.item())
             v_loss = np.nanmean(np.array(epoch_val_loss))
             writer.add_scalar('val/v_loss', v_loss, epoch)
             print('Epoch: {}/{} validate loss: {:.7f}'.format(epochs, epoch, v_loss))  # print loss for each epoch

    writer.close()  # close tensorboard


###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == "__main__":
    train_set = Dataset_Pro('D:/Datasets/pansharpening_2/training_data/train_wv3.h5')  # creat data for training
    training_data_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    validate_set = Dataset_Pro('D:/Datasets/pansharpening_2/valid_data/valid_wv3.h5')  # creat data for validation
    validate_data_loader = DataLoader(dataset=validate_set, num_workers=0, batch_size=batch_size, shuffle=False,
                                      pin_memory=True, drop_last=True)  # put training data to DataLoader for batches

    train(training_data_loader, validate_data_loader)  # call train function (call: Line 66)
    