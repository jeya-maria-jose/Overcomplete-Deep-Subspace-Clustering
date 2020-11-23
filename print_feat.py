import argparse
import torch
import torchvision
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import MNIST
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.utils import save_image
import torch
import cv2

parser = argparse.ArgumentParser(description='PyTorch_Siamese_lfw')
parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='number of total epochs to run(default: 1)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=16, type=int,
                    metavar='N', help='batch size (default: 8)')
parser.add_argument('--learning_rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--lfw_path', default='../lfw', type=str, metavar='PATH',
                    help='path to root path of lfw dataset (default: ../lfw)')
parser.add_argument('--train_list', default='../data/train.txt', type=str, metavar='PATH',
                    help='path to training list (default: ../data/train.txt)')
parser.add_argument('--test_list', default='../data/test.txt', type=str, metavar='PATH',
                    help='path to validation list (default: ../data/test.txt)')
parser.add_argument('--save_path', default='../data/', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ../data/)')
parser.add_argument('--aug', default='off', type=str,
                    help='turn on img augmentation (default: False)')
parser.add_argument('--cuda', default="off", type=str, 
                    help='switch on/off cuda option (default: off)')

parser.add_argument('--load', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--save', default='default', type=str,
                    help='turn on img augmentation (default: default)')
parser.add_argument('--model', default='autoencoder', type=str,
                    help='model name')


args = parser.parse_args()

location = '/home/jeyamariajose/Baselines/pytorch-beginner/08-AutoEncoder/sample/*.jpg'


def train_loader(path):
    img = Image.open(path)
    if args.aug != "off":
        pix = np.array(img)
        pix_aug = img_augmentation(pix)
        img = Image.fromarray(np.uint8(pix_aug))
    # print pix
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgshortList = []
            imgPath1, imgPath2, label = line.strip().split(' ')
            
            imgshortList.append(imgPath1)
            imgshortList.append(imgPath2)
            imgshortList.append(label)
            imgList.append(imgshortList)
    return imgList

class train_ImageList(data.Dataset):
    
    def __init__(self, fileList, transform=None, list_reader=default_list_reader, train_loader=train_loader):
        # self.root      = root
        self.imgList   = list_reader(fileList)
        self.transform = transform
        self.train_loader = train_loader

    def __getitem__(self, index):
        final = []
        [imgPath1, imgPath2, target] = self.imgList[index]
        img1 = self.train_loader(os.path.join(args.lfw_path, imgPath1))
        img2 = self.train_loader(os.path.join(args.lfw_path, imgPath2))

        # 
        # img2 = self.img_augmentation(img2)
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2, torch.from_numpy(np.array([target],dtype=np.float32))

    def __len__(self):
        return len(self.imgList)
        
# dataloader = torch.utils.data.DataLoader(
#                     train_ImageList(fileList=args.train_list, 
#                             transform=transforms.Compose([ 
#                             transforms.Scale((28,28)),
#                             transforms.ToTensor(),            ])),
#                     shuffle=False,
#                     num_workers=args.workers,
#                     batch_size=args.batch_size)

img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
num_epochs = 100
batch_size = 64
learning_rate = 1e-3
dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        self.encoder1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)  # b, 16, 10, 10
        self.encoder2=   nn.Conv2d(8, 16, 3, stride=1, padding=1)  # b, 8, 3, 3
        self.encoder3=   nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.decoder1 = nn.ConvTranspose2d(32, 16, 3, stride=2,padding=0)  # b, 16, 5, 5
        self.decoder2 =   nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1)  # b, 8, 15, 1
        self.decoder3 =   nn.ConvTranspose2d(8, 1, 4, stride=2, padding=0)  # b, 1, 28, 28
    

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.encoder1(x),2,2))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            cv2.imwrite("results/aelayer1_filter_{}.jpg".format(i),np.asarray(img))

        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))     
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            cv2.imwrite("results/aelayer2_filter_{}.jpg".format(i),np.asarray(img))   
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))        
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            cv2.imwrite("results/aelayer3_filter_{}.jpg".format(i),np.asarray(img))
        
        out = F.relu(self.decoder1(out))
        out = F.relu(self.decoder2(out))
        out = F.tanh(self.decoder3(out))
        return out

class rautoencoder(nn.Module):
    def __init__(self):        
        super(rautoencoder, self).__init__()
        
        self.encoder1 = nn.Conv2d(64, 32, 3, stride=2, padding=1)  # b, 16, 10, 10
            
        self.encoder2 = nn.Conv2d(32, 16, 3, stride=2, padding=1)  # b, 8, 3, 3
            
        self.encoder3 = nn.Conv2d(16, 1, 3, stride=2)  # b, 8, 3, 3

        self.decoder1 = nn.ConvTranspose2d(1, 16, 3, stride=2)  # b, 16, 5, 5
            
        self.decoder2 = nn.ConvTranspose2d(16, 32, 5, stride=2, padding=1)
        self.decoder3 = nn.ConvTranspose2d(32, 64, 2, stride=2, padding=1)  # b, 1, 28, 28
            

    def forward(self, x):
        # print(x.shape)
        out = F.relu(self.decoder1(x))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())

            # x = np.float32(src)
            # print(x.min(),x.max())
            # x = 255.0*(x-x.min())/(x.max()-x.min())
            # print(x.min(),x.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            
            cv2.imwrite("results/aelayer1_filter_{}.jpg".format(i),np.asarray(img))
        # print(out.shape)
        out = F.relu(self.decoder2(out))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            cv2.imwrite("results/aelayer2_filter_{}.jpg".format(i),np.asarray(img))
        # print(out.shape)
        out = F.relu(self.decoder3(out))
        for i in range(out.shape[1]):

            img = np.asarray(out[0][i].cpu().detach())
            img *= (255.0/img.max())
            img = cv2.applyColorMap(np.uint8(img),cv2.COLORMAP_JET)
            cv2.imwrite("results/aelayer3_filter_{}.jpg".format(i),np.asarray(img))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder1(out),1))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder2(out),1,1))        
        # print(out.shape)
        out = F.tanh(self.encoder3(out))        
        # print(out.shape)
        return out

model = rautoencoder().cuda()
model.load_state_dict(torch.load("./rautoencoder.pth"))
model.eval()

c =1
for data in dataloader:

        img = data[0]
        print(len(data))
        # print(len(img))
        # print(img.shape)
        img = torch.Tensor(img).cuda()
        print(img.shape)
        # ===================forward=====================
        output = model(img)
        print("done")
        c=c+1
        if c==4:
            break