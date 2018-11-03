import math

import torch
from utils.MyDataSet import MyDataSet
from torch.utils.data import DataLoader
import numpy as np
def predit(model,loder,name_str):
    print(name_str)
    mse = 0
    mae = 0
    for image, ground_truth in loder:
        density_map=model.forward(image,ground_truth)
        pre_sum = np.sum(density_map.data.cpu().numpy())
        gt_sum = np.sum(ground_truth.data.cpu().numpy())
        mae += abs(pre_sum - gt_sum)
        mse += abs(pre_sum - gt_sum) * abs(pre_sum - gt_sum)
    data_size=loder.__len__()
    print('Dataset Size:',data_size)
    mae/=data_size
    mse=math.sqrt(mse/data_size)
    print('MAE:',mae)
    print('MSE:',mse)


dirName=[['part_A','part_B'],['train_data','test_data']]
imageNumber=[300,182,400,316]


model=torch.load('model.pkl')
model.eval()
test_A_path="data/ShanghaiTech/"+dirName[0][0]+"/"+dirName[1][1]
test_A_data=MyDataSet(test_A_path,imageNumber[1],True)
test_A_loder=DataLoader(test_A_data,batch_size=1)
predit(model,test_A_loder,'Test A:')

test_B_path="data/ShanghaiTech/"+dirName[0][1]+'/'+dirName[1][1]
test_B_data=MyDataSet(test_B_path,imageNumber[3],True)
test_B_loader=DataLoader(test_B_data,batch_size=1)
predit(model,test_B_loader,'Test B:')



