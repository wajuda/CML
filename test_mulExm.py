import cv2
import torch.nn.modules as nn
import torch
from model import CMLNet
import h5py
import scipy.io as io
import os
import numpy as np

'''
def get_edge(data):  # for training
    rs = np.zeros_like(data)
    N = data.shape[0]
    for i in range(N):
        if len(data.shape) == 3:
            rs[i, :, :] = data[i, :, :] - cv2.boxFilter(data[i, :, :], -1, (5, 5))
        else:
            rs[i, :, :, :] = data[i, :, :, :] - cv2.boxFilter(data[i, :, :, :], -1, (5, 5))
    return rs
'''



def load_set(file_path):
    data = h5py.File(file_path)

    ms = data['ms']
    ms = (np.array(ms) / 2047.0)
    ms = ms.swapaxes(1, 3)
    ms = ms.swapaxes(2, 3)
    ms = torch.from_numpy(ms)


    pan = data['pan']  # HxW = 256x256
    pan = (np.array(pan) / 2047.0)

    pan = torch.from_numpy(pan)
    print(pan.shape)


    return  ms, pan

# ==============  Main test  ================== #



ckpt = "./pretrained.pth"#  12  1000 3.46

def test(file_path):
    ms, pan = load_set(file_path)

    model = CMLNet().cuda().eval()  # fixed, important!

    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight)  # fixed

    with torch.no_grad():
        for i in range(ms.shape[0]):

            x1, x2 =  ms[i, :, :, :], pan[i, :, :]   # read data: CxHxW (numpy type)

            x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
            x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1xCxHxW


            out2 = model(x1, x2)


            # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
            sr = torch.squeeze(out2).permute(1, 2, 0).cpu().detach().numpy()


            save_name = os.path.join("Result/WV3_Simu_mulExm1258", "{}.mat".format(i))  # fixed! save as .mat format that will used in Matlab!
            io.savemat(save_name, { "mulExm": sr})  # fixed!

if __name__ == '__main__':
    """@key: Absolute path"""

    file_path = "Dataset/WV3_Simu_mulExm/test1_mulExm1258.mat"
    test(file_path)
