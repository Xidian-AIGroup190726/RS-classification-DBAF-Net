import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import get_data
import torch.utils.data
from torch.utils.data import DataLoader
from libtiff import TIFF
import scipy.io as scio
import numpy as np
import cv2
from sklearn.metrics import confusion_matrix
import time

from torch.optim import lr_scheduler


class Bottleneck(nn.Module):

    def __init__(self, inplanes,outplanes, num_group=4, flag = 1):
        super(Bottleneck, self).__init__()
        self.flag = flag
        self.conv1 = nn.Conv2d(inplanes, int(outplanes/2), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(int(outplanes/2))
        self.conv2 = nn.Conv2d(int(outplanes/2), int(outplanes/2), kernel_size=3, stride=1,
                               padding=1, bias=False, groups=num_group)
        self.bn2 = nn.BatchNorm2d(int(outplanes/2))
        self.conv3 = nn.Conv2d(int(outplanes/2),outplanes , kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.flag == 1:
            out = self.relu(out)
        else:
            out = torch.sigmoid(out)

        return out
class DBAF(nn.Module):
    def __init__(self):
        super(DBAF,self).__init__()
        # self.patchsize = patchsize
        self.pre_ms = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(64),  # 64x64x64
            nn.ReLU(inplace=True)
        )  # 32x32x64
        self.pre_pan = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1, bias=False),  # (64+2*1-3)/1(向下取整)+1，size减半->112
            nn.BatchNorm2d(16),  # 64x64x64
            nn.ReLU(inplace=True)
        )
        self.mpool = nn.MaxPool2d(3,2,1)
        self.pan1 = Bottleneck(16,16)
        self.pan2_1 = Bottleneck(16,16)
        self.pan2_2 = Bottleneck(16,16)
        self.pan2_3 = Bottleneck(16,16)
        self.pan2_4 = Bottleneck(16,16,flag=2)
        self.pan2_5 = Bottleneck(16,16)
        self.pan3_1 = Bottleneck(32,32)
        self.pan3_2 = Bottleneck(48,48)
        self.pan3_3 = Bottleneck(48,32)
        self.pan3_4 = Bottleneck(32,32,flag=2)
        self.pan3_5 = Bottleneck(32,32)
        self.pan4_1 = Bottleneck(64,64)
        self.pan4_2 = Bottleneck(112,112)
        self.pan4_3 = Bottleneck(112,64)
        self.pan4_4 = Bottleneck(64,64,flag=2)
        self.pan4_5 = Bottleneck(64,64)
        self.ms1 = Bottleneck(64,64)
        self.fc1 = nn.Sequential(
            nn.Linear(64, 16, bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(16, 64, bias = True),
            nn.Sigmoid()
        )
        self.ms2 = Bottleneck(128,128)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 32, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(32, 128, bias=True),
            nn.Sigmoid()
        )
        self.ms3 = Bottleneck(128,128)
        self.fc3 = nn.Sequential(
            nn.Linear(256, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256, bias=True),
            nn.Sigmoid()
        )

        self.convp1 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(16, 16, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.convp2 = nn.Sequential(
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.convms31 =nn.Sequential(
            nn.Conv2d(256,64,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.bn = nn.BatchNorm2d(64)
        self.fconv1 = nn.Sequential(
            nn.Conv2d(64,64,3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.fconv2 = nn.Sequential(
            nn.Conv2d(128, 64, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.f1 = Bottleneck(64,64)
        self.f2 = Bottleneck(64,64)
        self.f3 = Bottleneck(64,64)
        self.f4 = Bottleneck(64,64,flag=2)
        self.f5 = Bottleneck(64,64)

        self.fc = nn.Sequential(
            nn.Linear(4800,800),
            nn.Linear(800,200),
            nn.Linear(200,10)
        )


    def forward(self, x1,x2):
        x1 = self.pre_ms(x1)  # 32x32x64
        x2 = self.pre_pan(x2)  # 32x32x16
        x1 = self.ms1(x1)#R
        x2 = self.pan1(x2) #R
        x31 = self.mpool(x2) #P
        x31  = self.pan2_1(x31)#R
        x32 = self.mpool(x31)#P
        x32  = self.pan2_2(x32)#R
        x33 =  F.interpolate(x32, size=(x31.shape[2], x31.shape[3]), mode='bilinear', align_corners=False)
        x33 = x31 + x33
        x33 = self.pan2_3(x33)
        x33 = F.interpolate(x33, size=(x2.shape[2], x2.shape[3]), mode='bilinear', align_corners=False)
        x33 = self.pan2_4(x33)
        x33 = x33 * x2
        x33 = x33 + x2
        x33 = self.pan2_5(x33)
        x33 = self.convp1(x33)

        m21 = F.avg_pool2d(x1, x1.shape[2])
        m21 = m21.view(m21.size(0), -1)
        m21 = self.fc1(m21)
        m21 = m21.reshape(m21.shape[0], m21.shape[1], 1, 1)
        m21 = m21 * x1
        m21 = m21 + x1
        m21 = self.ms1(m21)

        m30 = torch.cat((m21,x1),1)
        m31 = F.avg_pool2d(m30, m30.shape[2])
        m31 = m31.view(m31.size(0),-1)
        m31 = self.fc2(m31)
        m31 = m31.reshape(m31.shape[0], m31.shape[1], 1, 1)
        m31 = m31 * m30
        m31 = m31 + m30
        m31 = self.ms2(m31)

        x40 = torch.cat((x33,x31),1)
        x41 = self.mpool(x40)
        x41 = self.pan3_1(x41)
        x42 = torch.cat((x41,x32),1)
        x43 = self.mpool(x42)
        x43 = self.pan3_2(x43)
        x44 = F.interpolate(x43, size=(x42.shape[2], x42.shape[3]), mode='bilinear', align_corners=False)
        x44 = x44 + x42
        x44 = self.pan3_3(x44)
        x44 = F.interpolate(x44, size=(x40.shape[2], x40.shape[3]), mode='bilinear', align_corners=False)
        x44 = self.pan3_4(x44)
        x44 = x44 * x40
        x44 = x44 + x40
        x44 = self.convp2(x44)

        m40 = torch.cat((m31, m30), 1)
        m41 = F.avg_pool2d(m40, m40.shape[2])
        m41 = m41.view(m41.size(0), -1)
        m41 = self.fc3(m41)
        m41 = m41.reshape(m41.shape[0], m41.shape[1], 1, 1)
        m41 = m41 * m40
        m41 = self.convms31(m41)
        m42 = self.convms31(m40)

        x50 = torch.cat((x44,x41),1)
        x51 = self.mpool(x50)
        x51 = self.pan4_1(x51)
        x52 = torch.cat((x51, x43), 1)
        x53 = self.mpool(x52)
        x53 = self.pan4_2(x53)
        x53 = F.interpolate(x53, size=(x52.shape[2], x52.shape[3]), mode='bilinear', align_corners=False)
        x53 = x53 + x52
        x53 = self.pan4_3(x53)
        x53 = F.interpolate(x53, size=(x50.shape[2], x50.shape[3]), mode='bilinear', align_corners=False)
        x53 = self.pan4_4(x53)
        x53 = x53 * x50
        x53 = x53 + x50
        x53 = self.pan4_5(x53)

        xf1 = x53+m41
        xf1 = self.bn(xf1)
        xf1 = xf1 + x50
        xf1 = self.fconv1(xf1)
        xf1 = torch.cat((xf1,m42),1)
        xf1 = self.fconv2(xf1)

        xf2 = self.mpool(xf1)
        xf2 = self.f1(xf2)
        xf3 = F.avg_pool2d(xf2, xf2.shape[2])
        xf3 = xf3.view(xf3.size(0), -1)
        xf3 = self.fc1(xf3)
        xf3 = xf3.reshape(xf3.shape[0], xf3.shape[1], 1, 1)
        xf3 = xf3 * xf2
        xf3 = xf3 + xf2

        xf4 = self.mpool(xf3)
        xf4 = self.f2(xf4)
        xf5 = F.avg_pool2d(xf4,xf4.shape[2])
        xf5 = xf5.view(xf5.size(0),-1)
        xf5 = self.fc1(xf5)
        xf5 = xf5.reshape(xf5.shape[0], xf5.shape[1], 1, 1)
        xf5 = xf5 * xf4
        xf5 = xf5 + xf4

        xf6 = F.interpolate(xf5, size=(xf3.shape[2], xf3.shape[3]), mode='bilinear', align_corners=False)

        xf6 = xf6 + xf3
        xf6 = self.f3(xf6)
        xf7 = F.avg_pool2d(xf6, xf6.shape[2])
        xf7 = xf7.view(xf7.size(0), -1)
        xf7 = self.fc1(xf7)
        xf7 = xf7.reshape(xf7.shape[0], xf7.shape[1], 1, 1)
        xf7 = xf6 * xf7
        xf7 = xf6 + xf7

        xf7 = F.interpolate(xf7, size=(xf1.shape[2], xf1.shape[3]), mode='bilinear', align_corners=False)
        xf7 = self.f4(xf7)
        xf7 = xf7 * xf1
        xf7 = xf7 + xf1
        xf7 = self.f5(xf7)
        xf8 = F.avg_pool2d(xf7, xf7.shape[2])
        xf8 = xf8.view(xf8.size(0),-1)
        xf8 = self.fc1(xf8)
        xf8 = xf8.reshape(xf8.shape[0], xf8.shape[1], 1, 1)
        xf8 = xf8 * xf7
        xf8 = xf8 + xf7
        k1 = 3
        k2 = 3
        k3 = 3
        r1 = F.max_pool2d(xf8, kernel_size=k1, stride=k1, padding=0)
        r2 = F.max_pool2d(xf8, kernel_size=k2, stride=k2, padding=0)
        r3 = F.max_pool2d(xf8, kernel_size=k3, stride=k3, padding=0)
        r1 = r1.view(r1.size(0),-1)
        r2 = r2.view(r1.size(0),-1)
        r3 = r3.view(r1.size(0),-1)

        r = torch.cat((r1,r2,r3),1)

        x = self.fc(r)
        return x

if __name__ == "__main__":
    pan = torch.randn(2, 1, 64, 64)
    ms = torch.randn(2, 4, 16, 16)
    grf_net = DBAF()
    # out_result,coefxy = grf_net(ms,pan)
    # out_result,SSIM = grf_net(ms,pan)
    # 输入为MS和PAN
    out_result = grf_net(ms,pan)
    print(out_result)
    print(out_result.shape)

