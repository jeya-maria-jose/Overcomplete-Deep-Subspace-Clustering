__author__ = 'SherlockLiao'

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
from scipy.io import loadmat
import torch.utils.data as data_utils
import numpy as np

# x = loadmat("./Data/COIL20.mat")

# # print(x)
# feat = torch.from_numpy(x['fea'])
# gt = torch.from_numpy(x['gnd'])
# feat = np.reshape(feat,(1440,32,32,1))
# feat = feat.permute(0,3,2,1)
# print(feat.shape)
# train = data_utils.TensorDataset(feat, gt)
# dataloader = data_utils.DataLoader(train, batch_size=50, shuffle=False)

print("Begin")

direc = './results'
if not os.path.exists(direc):
    os.mkdir(direc)
    print("Folder Created")


def to_img(x):
    # x = 0.5 * (x + 1)
    # x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 32, 32)
    return x

num_epochs = 100
batch_size = 64
learning_rate = 1e-3


def add_noise(img):
    noise = torch.randn(img.size()) * 0.01
    noisy_img = img + noise
    return noisy_img



img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


dataset = MNIST('./data', transform=img_transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)


print("data loaded")

def custom_viz(kernels, path=None, cols=None):

    def set_size(w,h, ax=None):
        if not ax: 
            ax=plt.gca()
        l = ax.figure.subplotpars.left
        r = ax.figure.subplotpars.right
        t = ax.figure.subplotpars.top
        b = ax.figure.subplotpars.bottom
        figw = float(w)/(r-l)
        figh = float(h)/(t-b)
        ax.figure.set_size_inches(figw, figh)
    
    N = kernels.shape[0]
    C = kernels.shape[1]

    Tot = N*C

    # If single channel kernel with HxW size,# plot them in a row.# Else, plot image with C number of columns.
    if C>1:
        columns = C
    elif cols==None:
        columns = N
    elif cols:
        columns = cols
    rows = Tot // columns 
    rows += Tot % columns

    pos = range(1,Tot + 1)

    fig = plt.figure(1)
    fig.tight_layout()
    k=0
    for i in range(kernels.shape[0]):
        for j in range(kernels.shape[1]):
            img = kernels[i][j]
            ax = fig.add_subplot(rows,columns,pos[k])
            ax.imshow(img, cmap='gray')
            plt.axis('off')
            k = k+1

    set_size(30,30,ax)
    if path:
        plt.savefig(path, dpi=100)
    
    # plt.show()

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
        out = F.relu(F.max_pool2d(self.encoder2(out),2,2))        
        out = F.relu(F.max_pool2d(self.encoder3(out),2,2))        
        
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
        # print(out.shape)
        out = F.relu(self.decoder2(out))
        # print(out.shape)
        out = F.relu(self.decoder3(out))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder1(out),1))
        # print(out.shape)
        out = F.relu(F.max_pool2d(self.encoder2(out),1,1))        
        # print(out.shape)
        out = F.tanh(self.encoder3(out))        
        # print(out.shape)
        return out


model = autoencoder().cuda().float()
criterion = nn.MSELoss()
# print(model.parameters())
optimizer = torch.optim.Adam(list(model.parameters()), lr=learning_rate,
                             weight_decay=1e-5)
tm =0
# print(model.layers[0])
for epoch in range(100):
    # break
    for data in dataloader:
        img, _ = data
        # img = img.view(img.size(0), -1)
        noisy_img = add_noise(img)
        noisy_img = Variable(noisy_img).cuda().float()
        img = Variable(img).cuda()
        # ===================forward=====================
        
        output = model(img).float()
        # print(noisy_img.type,img.type,output.type)
        loss = criterion(output, img)
        # ===================backward====================
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 90:
            tm = tm +1
            if tm==20:
                imgg = img
                outputg = output
    
        # ===================log========================
    print('epoch [{}/{}], loss:{:.4f}'
          .format(epoch+1, num_epochs, loss.data))
    if epoch == 90 :
        tm = 0
        pic = to_img(outputg.cpu().data)
        save_image(pic, direc+'/outputa{}.png'.format(epoch))
        save_image(imgg, direc+'/input{}.png'.format(epoch))
        print("Image Saving")
        
        # kernels = model.encoder1.weight.cpu().detach().clone()
        # kernels = kernels - kernels.min()
        # kernels = kernels / kernels.max()
        # custom_viz(kernels, direc+'/en1_weights_{}.png'.format(epoch), 4)
        # kernels1 = model.encoder2.weight.cpu().detach().clone()
        # kernels1 = kernels1 - kernels1.min()
        # kernels1 = kernels1 / kernels1.max()
        # custom_viz(kernels1, direc+'/en2_weights_{}.png'.format(epoch), 4)
        # kernels2 = model.decoder2.weight.cpu().detach().clone()
        # kernels2 = kernels2- kernels2.min()
        # kernels2 = kernels2 / kernels2.max()
        # custom_viz(kernels2, direc+'/de1_weights_{}.png'.format(epoch), 4)
        # kernels3 = model.decoder2.weight.cpu().detach().clone()
        # kernels3 = kernels3 - kernels3.min()
        # kernels3 = kernels3 / kernels3.max()
        # custom_viz(kernels3, direc+'/de2_weights_{}.png'.format(epoch), 4)
        # kernels4 = model.decoder3.weight.cpu().detach().clone()
        # kernels4 = kernels4 - kernels4.min()
        # kernels4 = kernels4 / kernels4.max()
        # custom_viz(kernels4, direc+'/de3_weights_{}.png'.format(epoch), 4)
        

    torch.save(model.state_dict(), './autoencoder.pth')

