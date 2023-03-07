from torch.optim.lr_scheduler import MultiStepLR

# from .models import SegmentNet, DecisionNet, weights_init_normal
from dataset import KolektorDataset
from nets import DefectDetNet

import torch.nn as nn
import torch
import logging

from torchvision import datasets
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import os
import sys
import argparse
import time
import PIL.Image as Image
import numpy as np
import cv2
from criterion import IoU

# torch.set_printoptions(profile="full")

logging.basicConfig(filename="train_segment.txt", format="%(asctime)s : %(message)s", level=logging.INFO)

parser = argparse.ArgumentParser()

parser.add_argument("--cuda", type=bool, default=True, help="number of gpu")
parser.add_argument("--gpu_num", type=int, default=1, help="number of gpu")
parser.add_argument("--worker_num", type=int, default=0, help="number of input workers")
parser.add_argument("--batch_size", type=int, default=4, help="batch size of input")
parser.add_argument("--lr", type=float, default=0.1, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

parser.add_argument("--begin_epoch", type=int, default=0, help="begin_epoch")
parser.add_argument("--end_epoch", type=int, default=201, help="end_epoch")

parser.add_argument("--need_test", type=bool, default=True, help="need to test")
parser.add_argument("--test_interval", type=int, default=10, help="interval of test")
parser.add_argument("--need_save", type=bool, default=True, help="need to save")
parser.add_argument("--save_interval", type=int, default=10, help="interval of save weights")

parser.add_argument("--img_height", type=int, default=704, help="size of image height")
parser.add_argument("--img_width", type=int, default=256, help="size of image width")

opt = parser.parse_args()

print(opt)

dataSetRoot = "./Data"  # "/home/sean/Data/KolektorSDD_sean"  #

# ***********************************************************************

# Build nets
# segment_net = SegmentNet(init_weights=True)

model = DefectDetNet(3, 1)

# Loss functions
criterion_BCE = torch.nn.BCELoss()
criterion_IoU = IoU()

def loss_function(imgPredict, imgLabel, cls_pred, cls, mode):
    assert mode in {'Train', 'Test'}

    if mode == 'Train':
        # cls_loss = cls - cls_pred
        # cls_loss = abs(cls_loss)
        # cls_loss = torch.mean(cls_loss)
        cls_loss = criterion_BCE(cls_pred, cls)
    else:
        if cls:
            cls_loss = criterion_BCE(torch.tensor(np.ones((1, 1), dtype=np.float32)).cuda(), cls_pred)
        else:
            cls_loss = criterion_BCE(torch.tensor(np.zeros((1, 1), dtype=np.float32)).cuda(), cls_pred)
    # print(cls_loss)
    # print(criterion_IoU(imgPredict, imgLabel))
    # print(criterion_BCE(imgPredict, imgLabel))
    criterion_segment = criterion_BCE(imgPredict, imgLabel) + 0.1 * criterion_IoU(imgLabel, imgPredict) + cls_loss
    # criterion_segment = criterion_BCE(imgPredict, imgLabel) + cls_loss
    # criterion_segment = cls_loss
    # criterion_segment = criterion_BCE(imgPredict, imgLabel)
    return criterion_segment


logging.info("[loss: %s + %s + %s]" % (str(criterion_BCE), str(criterion_IoU), 'cls_loss'))
# logging.info("[loss: %s]" %(str(criterion_BCE)))

if opt.cuda:
    model = model.cuda()
    # criterion_segment.cuda()

if opt.gpu_num > 1:
    segment_net = torch.nn.DataParallel(model, device_ids=list(range(opt.gpu_num)))

if opt.begin_epoch != 0:
    # Load pretrained models
    model.load_state_dict(torch.load("./saved_models/segment_net_%d.pth" % (opt.begin_epoch)))
# else:
#     # Initialize weights
#     segment_net.apply(weights_init_normal)

# Optimizers
optimizer_seg = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_seg = torch.optim.ASGD(segment_net.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer_seg, milestones=[30, 50, 70], gamma=0.1)

transforms_ = transforms.Compose([
    transforms.Resize((opt.img_height, opt.img_width), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transforms_mask = transforms.Compose([
    transforms.Resize((opt.img_height // 8, opt.img_width // 8)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainOKloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask, subFold="Train_OK",
                    isTrain=True),
    batch_size=int(opt.batch_size / 2),
    shuffle=True,
    num_workers=opt.worker_num,
)

trainNGloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask, subFold="Train_NG",
                    isTrain=True),
    batch_size=int(opt.batch_size / 2),
    shuffle=True,
    num_workers=opt.worker_num,
)

'''
trainloader =  DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_,  transforms_mask= transforms_mask, subFold="Train_ALL", isTrain=True),
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.worker_num,
)
'''

testloader = DataLoader(
    KolektorDataset(dataSetRoot, transforms_=transforms_, transforms_mask=transforms_mask, subFold="Test",
                    isTrain=False),
    batch_size=1,
    shuffle=False,
    num_workers=opt.worker_num,
)


def show_img(image):
    for num in range(image.shape[0]):
        print(str(num) + "th image in " + str(image.shape[0]) + " images")
        img = image[num].cpu().detach().numpy()
        img = np.transpose(img, (1, 2, 0))  # C*H*W -> H*W*C
        # plt.imshow(img, cmap='gray')
        # plt.show()
        cv2.imshow('1', img)
        cv2.waitKey()


for epoch in range(opt.begin_epoch, opt.end_epoch):

    iterOK = trainOKloader.__iter__()
    iterNG = trainNGloader.__iter__()

    lenNum = min(len(trainNGloader), len(trainOKloader))
    lenNum = lenNum - 1

    model.train()
    # train *****************************************************************
    for i in range(0, lenNum):
        batchData_OK = iterOK.__next__()
        # idx, batchData = enumerate(trainOKloader)
        batchData_NG = iterNG.__next__()
        # idx, batchData = enumerate(trainNGloader)
        batchData = {"img": torch.cat((batchData_OK["img"], batchData_NG["img"]), 0),
                     "mask": torch.cat((batchData_OK["mask"], batchData_NG["mask"]), 0)}

        # batchData = {"img": batchData_NG["img"], "mask": batchData_NG["mask"]}

        if opt.cuda:
            img = batchData["img"].cuda()
            mask = batchData["mask"].cuda()
        else:
            img = batchData["img"]
            mask = batchData["mask"]

        # show_img(mask)
        optimizer_seg.zero_grad()

        rst = model(img)

        # seg = torch.round(rst["seg"])
        seg = rst["seg"]
        cls = rst["cls"]
        # mask = torch.ceil(mask)
        # if epoch >= 10:
        #     show_img(seg)
        #     show_img(mask)
        # print(torch.max(seg))
        # print(torch.max(mask))

        target = torch.cat((torch.zeros((int(opt.batch_size / 2), 1)), torch.ones((int(opt.batch_size / 2), 1)))).cuda()
        if epoch <= 20:
            loss_seg = criterion_BCE(seg, mask) + 0.1*criterion_IoU(seg, mask)
        else:
            loss_seg = loss_function(seg, mask, cls, target, 'Train')


        loss_seg.backward()
        optimizer_seg.step()

        sys.stdout.write(
            "\r [Epoch %d/%d]  [Batch %d/%d] [loss %f]"
            % (
                epoch,
                opt.end_epoch - 1,
                i,
                lenNum - 1,
                loss_seg.item()
            )
        )
    # test ****************************************************************************
    if opt.need_test and epoch % opt.test_interval == 0 and epoch >= opt.test_interval:
        model.eval()
        loss_epoch = []
        # segment_net.eval()
        with torch.no_grad():
            for i, testBatch in enumerate(testloader):
                imgTest = testBatch["img"].cuda()
                maskTest = testBatch["mask"].cuda()
                cls = testBatch["label"]

                t1 = time.time()
                rstTest = model(imgTest)
                t2 = time.time()
                # segTest = torch.round(rstTest["seg"])
                segTest = rstTest["seg"]
                cls_pred = rstTest["cls"]
                # maskTest = torch.ceil(maskTest)
                testLoss = loss_function(segTest, maskTest, cls_pred, cls, 'Test')

                loss_epoch.append(testLoss.cpu().detach().numpy())

                save_path_str = "./testResultSeg/epoch_%d" % epoch
                if os.path.exists(save_path_str) == False:
                    os.makedirs(save_path_str, exist_ok=True)
                    # os.mkdir(save_path_str)

                # print("processing image NO %d, time comsuption %fs" % (i, t2 - t1))
                save_image(imgTest.data, "%s/img_%d.jpg" % (save_path_str, i))
                save_image(segTest.data, "%s/img_%d_seg.jpg" % (save_path_str, i))

            logging.info("\r [Epoch %d/%d] [loss %f] [lr %f]" % (
                epoch, opt.end_epoch, np.mean(loss_epoch), optimizer_seg.state_dict()['param_groups'][0]['lr']))
    scheduler.step()

    # save parameters *****************************************************************
    if opt.need_save and epoch % opt.save_interval == 0 and epoch >= opt.save_interval:
        # segment_net.eval()

        save_path_str = "./saved_models"
        if os.path.exists(save_path_str) == False:
            os.makedirs(save_path_str, exist_ok=True)

        torch.save(model.state_dict(), "%s/DefectDetNet_%d.pth" % (save_path_str, epoch))
        print("save weights ! epoch = %d" % epoch)
        # segment_net.train()
        pass
