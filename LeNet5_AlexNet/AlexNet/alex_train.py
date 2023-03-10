#alex_train
import torch
from torch import nn
from alex_net import MyAlexNet
import numpy as np
from torch.optim import lr_scheduler
import os
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#训练集的路径
ROOT_TRAIN = r'D:/AStudy/python_study/CNN/AlexNet/data/train'
#测试集的路径
ROOT_TEST = r'D:/AStudy/python_study/CNN/AlexNet/data/val'

#将图像的像素值归一化到[0,1]之间
normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

#来对图像就行变化，处理数据，裁剪等操作，
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(),#随机地对图像进行垂直翻转，以增加数据多样性。
    transforms.ToTensor(),#将图像转换为张量，并将像素值归一化到 $[0,1]$ 范围内。
    normalize])#使用给定的均值和标准差对图像进行标准化，以使得每个通道的均值为0，标准差为1。

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize])

train_dataset = ImageFolder(ROOT_TRAIN, transform= train_transform)
val_dataset = ImageFolder(ROOT_TEST, transform= val_transform)

train_dataloader = DataLoader(train_dataset, batch_size= 32, shuffle= True)
val_dataloader = DataLoader(val_dataset, batch_size= 32, shuffle= True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#将模型输入到设备中训练
model = MyAlexNet().to(device)

#定义损失函数
loss_fn = nn.CrossEntropyLoss()
#定义优化器
optimizer = torch.optim.SGD(model.parameters(), lr= 0.01, momentum= 0.9)

#学习率每隔10轮变为原来的0.5
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.5)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (x, y) in enumerate(dataloader):
        image, y = x.to(device), y.to(device)
        output = model(image)
        cur_loss = loss_fn(output, y)
        _, pred = torch.max(output, axis= 1)
        cur_acc = torch.sum(y == pred)/output.shape[0]

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()
        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1

    train_loss = loss/n
    train_acc = current/n

    print('train_loss: ' + str(train_loss))
    print('train_acc: ' + str(train_acc))
    return train_loss, train_acc


# 定义验证函数
def val(dataloader, model, loss_fn):
    #将模型转化为验证模型
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():
        for batch, (x, y) in enumerate(dataloader):
            image, y = x.to(device), y.to(device)
            output = model(image)
            cur_loss = loss_fn(output, y)
            _, pred = torch.max(output, axis=1)
            cur_acc = torch.sum(y == pred) / output.shape[0]

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1

    val_loss = loss / n
    val_acc = current / n

    print('val_loss: ' + str(val_loss))
    print('val_acc: ' + str(val_acc))
    return val_loss, val_acc

#定义loss画图函数
def matplot_loss(train_loss, val_loss):
    plt.plot(train_loss, label= 'train_loss')
    plt.plot(val_loss, label = 'val_loss')
    plt.legend(loc = 'best')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.title('训练集和测试集loss值进行对比')
    plt.show()

#定义acc画图函数
def matplot_acc(train_acc, val_acc):
    plt.plot(train_loss, label= 'train_acc')
    plt.plot(val_loss, label = 'val_acc')
    plt.legend(loc = 'best')
    plt.ylabel('acc')
    plt.xlabel('epoch')
    plt.title('训练集和测试集acc值进行对比')
    plt.show()

#开始训练
#数组用来画图
loss_train = []
acc_train = []
loss_val = []
acc_val = []

epoch = 20
max_acc = 0.0
for i in range(epoch):
    lr_scheduler.step()
    print(f"epoch{i+1}--------------------")
    train_loss, train_acc = train(train_dataloader, model, loss_fn, optimizer)
    val_loss, val_acc = val(val_dataloader, model, loss_fn)

    loss_train.append(train_loss)
    acc_train.append(train_acc)
    loss_val.append(val_loss)
    acc_val.append(val_acc)

    #保存最好的模型权重
    if val_acc > max_acc:
        max_acc = val_acc
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        print(f"save best model, 第{i+1}轮")
        torch.save(model.state_dict(), 'save_model/best_model.pth')

    #保存最后一轮的权重文件
    if i == epoch - 1:
        torch.save(model.state_dict(), 'save_model/last_model.pth')

#画图
matplot_loss(loss_train, loss_val)
matplot_acc(acc_train, acc_val)
print('done')



