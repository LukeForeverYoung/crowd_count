import torch
import torchvision
from utils.MyDataSet import MyDataSet
from torch.utils.data import DataLoader
from utils.Model import *
import numpy as np
import  matplotlib.pyplot as plt
#Hyper Parameter
lr=0.00001
Batch_size=16
epoch=2000
dirName=[['part_A','part_B'],['train_data','test_data']]
imageNumber=[300,182,400,316]
path="data/ShanghaiTech/"+dirName[0][0]+"/"+dirName[1][0]
train_A=MyDataSet(path,int(imageNumber[0]),True)
collate_old = torch.utils.data.dataloader.default_collate


train_A_lodaer=DataLoader(train_A,batch_size=1)

model=Crowd_count_model()
model.cuda()
model.train()
for p in model.parameters():
    p.requires_grad=True
model.apply(weight_normal_init)
print('weight init ok')
optimizer=torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)


'''test
for i in range(100000):
    (im, gt) = train_A.__getitem__(0)
    im = im[None, :, :]
    gt = gt[None, :, :]
    den=model.forward(im, gt)
    loss = model.loss_mse
    #print(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if(i%300==0):
        plt.imshow(den.data.cpu().numpy().squeeze())
        plt.show()

input()
'''
def test_show(den,gt):
    plt.ion()
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(den)
    plt.subplot(1,2,2)
    plt.imshow(gt)
    plt.pause(3)
    plt.close()


for i in range(epoch):
    train_loss=0
    step=-1
    for image,ground_truth in train_A_lodaer:
        #print(image.shape,'\n',ground_truth.shape)
        den=model.forward(image,ground_truth)
        loss=model.loss_mse
        train_loss+=loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print('step ok')
        if(step%100==0):
            pass
            #print(train_loss)
            #test_show(den.data.cpu().numpy().squeeze(),ground_truth.data.cpu().numpy().squeeze())
            #train_loss=0
        step+=1
    print(i,'/',epoch,'loss:',train_loss)
    train_loss=0

print('train finish!')
path="data/ShanghaiTech/"+dirName[0][0]+"/"+dirName[1][1]
test_A=MyDataSet(path,int(imageNumber[1]),True)
test_A_lodaer=DataLoader(test_A,batch_size=1)
train_loss=0
for image, ground_truth in train_A_lodaer:
    # print(image.shape,'\n',ground_truth.shape)

    density_map=model.forward(image, ground_truth)
    pre_sum=np.sum(density_map.data.cpu().numpy())
    gt_sum=np.sum(ground_truth.data.cpu().numpy())
    train_loss += abs(pre_sum-gt_sum)*abs(pre_sum-gt_sum)
print(train_loss/imageNumber[1])
torch.save(model,'model.pkl')