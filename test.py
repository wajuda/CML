# This is a pytorch version for the work of PanNet
# YW Jin, X Wu, LJ Deng(UESTC);
# 2020-09;


import torch


from model import CMLNet
import h5py
import scipy.io as sio
import os


def load_set(file_path):

    data = h5py.File(file_path)
    ms= torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    pan = torch.from_numpy(data['pan'] / 2047.0)   # HxW = 256x256

    return ms, pan


def Dataset_Pro(file_path):

        data = sio.loadmat(file_path)
        ms =(data['ms'] / 2047.0).transpose(2, 0, 1)  # CxHxW= 8x64x64
        ms= torch.from_numpy(ms)
        pan =(data['pan'] / 2047.0)  # HxW = 256x256
        pan = torch.from_numpy(pan)

        return ms, pan
###################################################################
# ------------------- Main Test (Run second) -------------------
###################################################################

ckpt = "./Weights/wv3_newmulti/pretrained.pth"#"Weights/wv3/400.pth"

def test(file_path):
    ms, pan = Dataset_Pro(file_path)

    model = CMLNet().cuda().eval()   # fixed, important!

    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():

        x1, x2 = ms, pan    # read data: CxHxW (numpy type)
        print(x1.shape)

        x1 = x1.cuda().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))

        x2 = x2.cuda().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW
        with torch.autograd.profiler.profile(enabled=True) as prof:
            sr = model(x1, x2)  # tensor type: CxHxW
        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

        save_name = os.path.join("01-WV_DataTest/01-All_Comparisons/2_DL_Result/Our/WV3_Simu_cml/",
                                     "new_data6.mat")  # fixed! save as .mat format that will used in Matlab!

        sio.savemat(save_name, {'result': sr})  # fixed!
###################################################################
# ------------------- Main Function (Run first) -------------------
###################################################################
if __name__ == '__main__':
    file_path = "01-WV_DataTest/01-All_Comparisons/1_TestData/WV3_Simu/new_data6.mat"
    test(file_path)   # recall test function
