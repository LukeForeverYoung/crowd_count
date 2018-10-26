import torch
import torchvision
from utils.MyDataSet import MyDataSet
from torch.utils.data import DataLoader
#Hyper Parameter
Batch_size=16
epoch=1
dirName=[['part_A','part_B'],['train_data','test_data']]
imageNumber=[300,182,400,316]
path="data/ShanghaiTech/"+dirName[0][0]+"/"+dirName[1][0]
train_A=MyDataSet(path,int(imageNumber[0]/10),True)
collate_old = torch.utils.data.dataloader.default_collate


train_A_lodaer=DataLoader(train_A,batch_size=1)
for i in range(epoch):
    for image,ground_truth in train_A_lodaer:
        print(image.shape,'\n',ground_truth.shape)
        input()