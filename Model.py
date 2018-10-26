import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        #
        self.Column_1=nn.Sequential(
            Conv2dSeq(in_channels=1,out_channels=16,kernel_size=(9,9)),
            nn.MaxPool2d(kernel_size=2)
        )


#conv->activate->bn
class Conv2dSeq(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride=1,same_padding=False,bn=False):
        super(Conv2dSeq,self).__init__()
        padding=0
        if(same_padding):
            padding=int((kernel_size-1)/2)
        self.conv=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride)
        self.bn=None
        if(bn):
            self.bn=nn.BatchNorm2d(out_channels,eps=0.001,momentum=0,affine=True)
        self.relu=nn.ReLU(inplace=True)

    def forward(self, input):
        input=self.conv(input)
        if(self.bn):
            input=self.bn(self)
        if(self.relu):
            input=self.relu(input)
        return input
    


