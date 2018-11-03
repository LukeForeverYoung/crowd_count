import torch
import torch.nn as nn
#conv->activate->bn
class Conv2dSeq(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,same_padding=True,bn=False):
        super(Conv2dSeq,self).__init__()
        padding=0
        if(same_padding):
            padding=int((kernel_size-1)/2)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn=None
        if(bn):
            self.bn=nn.BatchNorm2d(out_channels,eps=0.001,momentum=0,affine=True)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        input=self.conv(input)
        if(self.bn):
            input=self.bn(input)
        if(self.relu):
            input=self.relu(input)
        return input


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #
        self.Column_1=nn.Sequential(
            Conv2dSeq(in_channels=1,out_channels=16,kernel_size=9),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=16,out_channels=32,kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=32,out_channels=16,kernel_size=7),
            Conv2dSeq(in_channels=16, out_channels=8, kernel_size=7),
        )
        self.Column_2 = nn.Sequential(
            Conv2dSeq(in_channels=1, out_channels=20, kernel_size=7),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=20, out_channels=40, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=40, out_channels=20, kernel_size=5),
            Conv2dSeq(in_channels=20, out_channels=10, kernel_size=5),
        )
        self.Column_3 = nn.Sequential(
            Conv2dSeq(in_channels=1, out_channels=24, kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=24, out_channels=48, kernel_size=3),
            nn.MaxPool2d(kernel_size=2),
            Conv2dSeq(in_channels=48, out_channels=24, kernel_size=3),
            Conv2dSeq(in_channels=24, out_channels=12, kernel_size=3),
        )
        self.fuse=nn.Sequential(
            Conv2dSeq(in_channels=30, out_channels=1, kernel_size=1),
        )

    def forward(self, input):
        c1=self.Column_1(input)
        c2=self.Column_2(input)
        c3=self.Column_3(input)
        #shape (1,channels,height,width)
        #print(input.shape,c1.shape,c2.shape,c3.shape)
        merge=torch.cat((c1,c2,c3),1)
        output=self.fuse(merge)
        return output

class Crowd_count_model(nn.Module):
    def __init__(self):
        super(Crowd_count_model,self).__init__()
        self.network=Model()
        self.loss_function=nn.MSELoss()
    @property
    def loss(self):
        return self.loss_mse

    def forward(self, input,gt):
        density_map=self.network(input)
        if self.training:
            self.loss_mse=self.calculate_loss(density_map,gt)
        return density_map

    def calculate_loss(self,density_map,gt):
        return self.loss_function(density_map,gt)

def weight_normal_init(m):
    dev = 0.01
    if isinstance(m,nn.Conv2d):
        m.weight.data.normal_(0.0,dev)
        if(m.bias is not None):
            m.bias.data.fill_(0.0)
    elif  isinstance(m,nn.Linear):
        m.weight.data.normal_(0.0, dev)

def weight_test(model,dev=0.01):
    if isinstance(model, list):
        for m in model:
            weight_test(m, dev)
    else:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                # print torch.sum(m.weight)
                m.weight.data.normal_(0.0, dev)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, dev)



