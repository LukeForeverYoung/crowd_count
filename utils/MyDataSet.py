from torch.utils.data import Dataset
import torch
import torchvision as tv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class MyDataSet(Dataset):
    def __init__(self, file,image_number,is_cuda):
        self.transform = tv.transforms.Compose([tv.transforms.ToTensor()])
        self.number=image_number
        self.is_cuda=is_cuda
        self.images=[]
        self.ground_truths=[]
        self.file_name=[]
        for i in range(1,image_number+1):
            self.file_name.append(file+"/images/IMG_"+str(i)+".jpg")
            #print(file+"/images/IMG_"+str(i)+".jpg")
            image=plt.imread(file+"/images/IMG_"+str(i)+".jpg")
            #print(image.shape)
            if(image.ndim==3):# convert rgb to gray
                image=np.dot(image[:,:,:3],[0.299, 0.587, 0.114])
            (h,w)=image.shape
            image=image.reshape((h,w,1))
            #print(image.shape)
            image=image
            #image=transform(plt.imread(file+"/images/IMG_"+str(i)+".jpg"))
            #ground_truth=torch.from_numpy(np.load(file+"/ground-truth/Hot_IMG_"+str(i)+".npy")).float()
            ground_truth=np.load(file+"/ground-truth/Hot_IMG_"+str(i)+".npy")[:,:,None]
            self.images.append(image)
            self.ground_truths.append(ground_truth)

    def __getitem__(self, index):
        #print(self.file_name[index])
        image = self.transform(self.images[index]).float()
        ground_truth = self.transform(self.ground_truths[index]).float()
        if(self.is_cuda):
            image=image.cuda()
            ground_truth=ground_truth.cuda()

        #Build Tensor as late as possiable to avoid unnecessary gradient calculation
        return image, ground_truth

    def __len__(self):
        return self.number
