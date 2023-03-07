
import numpy as np
import torch
from torch import nn

__all__ = ['IoU']


class IoU(nn.Module):
    def __init__(self):
        super(IoU, self).__init__()

    def forward(self, imgPredict, imgLabel):
        # compute the IoU of the foreground
        Iand = torch.sum(imgLabel * imgPredict)
        Ior = torch.sum(imgLabel) + torch.sum(imgPredict) - Iand
        IoU = Iand / Ior

        # IoU loss is (1-IoU1)
        IoU = 1 - IoU

        return IoU



if __name__ == '__main__':
    # imgPredict = torch.tensor([[1, 1, 0, 0], [1, 1, 0, 0]])
    # imgLabel = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    imgPredict = torch.randint(0, 2, (4, 1, 4, 4))  # 可直接换成预测图片
    imgLabel = torch.randint(0, 2, (4, 1, 4, 4))  # 可直接换成标注图片

    metric = IoU()

    mIoU = metric.forward(imgPredict, imgLabel)
    print('mIoU is : %f' % mIoU)
