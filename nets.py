import torch.nn as nn
import torch
from torch import Tensor
import torch.nn.functional as F


def initialize(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight.data, 1)
        if module.bias is not None:
            nn.init.constant_(module.bias.data, 0)
    elif isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')


class DefectDetNet(nn.Module):
    def __init__(self, in_channels, cls_num=1):
        super(DefectDetNet, self).__init__()
        self.dec_cls_num = cls_num

        # Segment net part
        self.seg_layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.seg_layer1 = nn.Sequential(
            nn.Conv2d(32, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.seg_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.seg_layer3 = nn.Sequential(
            nn.Conv2d(64, 1024, 15, stride=1, padding=7),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True)
        )

        # ���չٷ�tensorflow��Ĵ��룬1x1�����û��ʹ��ReLu��ʹ����Sigmoid���ɶ�ֵ����
        # self.seg_layer4 = nn.Sequential(
        #     nn.Conv2d(1024, 1, 1),
        #     nn.BatchNorm2d(1),
        #     nn.Sigmoid()
        # )

        self.seg_layer4 = nn.Sequential(
            nn.Conv2d(1024, 1, 1),
            nn.BatchNorm2d(1)
            # nn.BatchNorm2d(1, bias=False)
        )

        # Decision net part
        self.dec_layer0 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(1027, 8, 5, stride=1, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dec_layer1 = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.dec_layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        # self.dec_avg32 = nn.AdaptiveAvgPool2d((1, 1))
        # self.dec_max32 = nn.AdaptiveMaxPool2d((1, 1))
        # self.dec_avg1 = nn.AdaptiveAvgPool2d((1, 1))
        # self.dec_max1 = nn.AdaptiveMaxPool2d((1, 1))

        # self.dec_fc = nn.Linear(70, self.dec_cls_num)
        self.seg_fc = nn.Linear(1025, cls_num)

    @staticmethod
    def global_avg_pool(x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = F.avg_pool2d(x, (h, w))
        return x.reshape(n, c)

    @staticmethod
    def global_max_pool(x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x = F.max_pool2d(x, (h, w))
        return x.reshape(n, c)

    def forward(self, x):
        # Segment
        seg_x = self.seg_layer0(x)
        seg_x = self.seg_layer1(seg_x)
        seg_x = self.seg_layer2(seg_x)
        features = self.seg_layer3(seg_x)
        mask = self.seg_layer4(features)
        mask = torch.sigmoid(mask)

        # # Decision
        # dec_x = torch.cat((features, torch.sigmoid(mask)), 1)
        # dec_x = self.dec_layer0(dec_x)
        # dec_x = self.dec_layer1(dec_x)
        # dec_x = self.dec_layer2(dec_x)
        #
        # dec_max = self.global_max_pool(dec_x)
        # dec_avg = self.global_avg_pool(dec_x)
        fea_max = self.global_max_pool(features)
        seg_max = self.global_max_pool(mask)
        # seg_avg = self.global_avg_pool(torch.sigmoid(mask))

        vector = torch.cat((fea_max, seg_max), 1)
        cls = torch.sigmoid(self.seg_fc(vector))

        return {"seg": mask, "cls": cls}


if __name__ == '__main__':
    net = DefectDetNet(3, 1)
    input = torch.randn((2, 3, 64, 64))
    output = net(input)
    mask, cls = output['seg'], output['cls']

    print(mask.shape)
    print(cls.shape)
