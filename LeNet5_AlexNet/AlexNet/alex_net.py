#alex_net
import torch
from torch import nn
import torch.nn.functional as F

class MyAlexNet(nn.Module):
    def __init__(self):
        super(MyAlexNet, self).__init__()
        # 激活函数
        self.ReLU = nn.ReLU()
        #网络的第一层卷积层，这一次是三通道的图片
        self.c1 = nn.Conv2d(in_channels= 3, out_channels= 48, kernel_size= 11, stride= 4, padding= 2)
        #第二层卷积
        self.c2 = nn.Conv2d(in_channels= 48, out_channels= 128, kernel_size= 5, stride= 1, padding= 2)
        #池化层
        self.s2 = nn.MaxPool2d(2)
        #第三层卷积
        self.c3 = nn.Conv2d(in_channels= 128, out_channels= 192, kernel_size= 3, stride= 1, padding = 1)
        # 池化层
        self.s3 = nn.MaxPool2d(2)
        # 第四层卷积
        self.c4 = nn.Conv2d(in_channels= 192, out_channels= 192, kernel_size= 3, stride= 1, padding= 1)
        # 第五层卷积
        self.c5 = nn.Conv2d(in_channels= 192, out_channels= 128, kernel_size= 3, stride= 1, padding= 1)
        #池化层
        self.s4 = nn.MaxPool2d(kernel_size= 3, stride= 2)
        #平展层
        self.flatten = nn.Flatten()
        #全连接层
        self.f6 = nn.Linear(4608, 2048)
        self.f7 = nn.Linear(2048, 2048)
        self.f8 = nn.Linear(2048, 1000)
        self.f9 = nn.Linear(1000, 2)

    def forward(self, x):
        x = self.ReLU(self.c1(x))
        x = self.ReLU(self.c2(x))
        x = self.s2(x)
        x = self.ReLU(self.c3(x))
        x = self.s3(x)
        x = self.ReLU(self.c4(x))
        x = self.ReLU(self.c5(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f6(x)
        x = F.dropout(x, p = 0.5)
        x = self.f7(x)
        x = F.dropout(x, p=0.5)
        x = self.f8(x)
        x = F.dropout(x, p=0.5)
        x = self.f9(x)

        return x

if __name__ == "__main__":
    x = torch.rand([1, 3, 224, 224])
    model = MyAlexNet()
    y = model(x)
    print(x)
    print(y)



