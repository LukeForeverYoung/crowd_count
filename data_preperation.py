from PIL import Image
import scipy.io as sio
import  matplotlib.pyplot as plt
import numpy as np
import csv
import math
import threading
from multiprocessing import Process,Pool

filter_size=15
beta=0.3
padding=int(filter_size/2)
k_close=7


def getGaussianFilter(size,sigma):
    mid=int(size/2)
    sum=0
    gaussian = np.zeros([size, size])
    for i in range(size):
        for j in range(size):
            gaussian[i, j] = math.exp(-1 / 2 * (np.square(i - mid) / np.square(sigma)  # 生成二维高斯分布矩阵
                                            + (np.square(j - mid) / np.square(sigma)))) / (2 * math.pi * sigma * sigma)
            sum = sum + gaussian[i, j]
    gaussian = gaussian / sum
    return gaussian


def distance(pointA,pointB):
    return math.sqrt((pointA[0]-pointB[0])*(pointA[0]-pointB[0])+(pointA[1]-pointB[1])*(pointA[1]-pointB[1]))


def kCloseMean(points,index,k):
    dMean=0
    stdp=points[index]
    points.remove(stdp)
    points.sort(key=lambda p:distance(p,stdp))
    count=0
    for point in points:
        dMean+=distance(point,stdp)
        count+=1
        if(count==k):
            break
    if(count==0):
        return 1
    return dMean*1.0/count

def floor_div4(v):
    return math.floor(math.floor(v/2)/2)


def solve(pName,dName,index):
    path = 'data/ShanghaiTech/' + pName + '/' + dName + '/'
    image = Image.open(path + 'images/IMG_' + str(index) + '.jpg')
    [w, h] = image.size
    with open(path + 'ground-truth/GT_IMG_' + str(index) + '.csv') as f:
        reader = csv.reader(f)
        data = list(reader)
        number = int(data[0][0])
        points = [[float(p[0])/4,float(p[1])/4] for p in data[1:]]
        nw=floor_div4(w)
        nh=floor_div4(h)
        #print(nw,nh)
        H_map = np.zeros((nh + padding * 2, nw + padding * 2))#原密度图先加入padding
        for j, point in enumerate(points):
            # print(i)
            y = int(point[1])
            x = int(point[0])
            sigma = beta * kCloseMean(points.copy(), j, k_close)

            #print(sigma)
            gaussian = getGaussianFilter(filter_size, sigma)
            #对于越界的区域,计算其有效值
            sum = 0.0
            for yy in range(filter_size):
                for xx in range(filter_size):
                    if (yy + y >= padding and yy + y < padding + nh and xx + x >= padding and xx + x < padding + nw):
                        sum = sum + gaussian[yy, xx]
            #处以有效值所占比例就可以让切割后的区域的积分维持在1,且中心依旧是label定位的点
            gaussian /= sum
            H_map[y:y + filter_size, x:x + filter_size] += gaussian[:, :]
            # print(sigma)
            # print(gaussian)
            # print(H_map[y:y+filter_size,x:x+filter_size])
        H_map = H_map[padding:padding + nh, padding:padding + nw]
        #print(H_map.sum(),number) #可以验证最终密度图的积分等于人数总数
        #plt.imshow(H_map)
        #plt.show()
        #保存numpy内容至二进制文件中
        np.save(path + 'ground-truth/Hot_IMG_' + str(index) + '.npy', H_map)
        print('ok', path + 'ground-truth/Hot_IMG_' + str(index) + '.npy')



if __name__ =='__main__':#进程不能共用内存
    dirName = [['part_A', 'part_B'], ['train_data', 'test_data']]
    imageNumber = [300, 182, 400, 316]

    dirIndex = 0
    process = Pool(5)
    for pName in dirName[0]:
        for dName in dirName[1]:
            for i in range(1,imageNumber[dirIndex]+1):
                #print(i)
                process.apply_async(func=solve, args=(pName, dName, i))
                #solve(pName,dName,i)
            dirIndex += 1
    process.close()
    process.join()

    print("Finish")


