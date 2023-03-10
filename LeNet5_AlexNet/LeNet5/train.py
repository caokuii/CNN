#lenet5_train
import os

import torch
from torch import nn
from net import MyLeNet5
from torch.optim import lr_scheduler
from torchvision import datasets, transforms

#数据转化为张量tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root='./data', train= True, transform= data_transform, download= True)
#个数据加载器，用于将数据集分成小批量逐个加载到模型中进行训练或推理
train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= 16, shuffle= True)

#加载测试集数据
test_dataset = datasets.MNIST(root='./data', train= False, transform= data_transform, download= True)
test_dataloader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= 16, shuffle= True)

#如果有显卡，可以转到GPU，因为我的电脑没有独立显卡所以选择直接在CPU上训练
device = "cuda" if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

#调用net里面定义的模型，将模型转到device中训练
#model = MyLeNet5()
model =  MyLeNet5().to(device)

#定义一个损失函数（交叉熵损失）
loss_fn = nn.CrossEntropyLoss()

#定义一个优化器
optimizer = torch.optim.SGD(model.parameters(), lr= 1e-3, momentum= 0.9)

#学习每隔10轮，变为原来的0.1
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size= 10, gamma= 0.1)

#定义训练函数
def train(dataloader, model, loss_fn, optimizer):
    loss, current, n = 0.0, 0.0, 0
    for batch, (X, y) in enumerate(dataloader):
        # 向前传播
        #X, y = X.to(device), y.to(device)
        output = model(X)
        cur_loss = loss_fn(output, y)#计算损失值，输出值与真实值进行交叉熵计算
        _, pred = torch.max(output, axis= 1)#输出每行最大的概率，意思是在10个概率中取最大的概率，然后取出概率对应最大的索引值

        cur_acc = torch.sum(y == pred)/output.shape[0]#计算批次的精确度

        #反向传播
        optimizer.zero_grad()
        cur_loss.backward()
        optimizer.step()    #梯度更新

        loss += cur_loss.item()
        current += cur_acc.item()
        n = n + 1
    #输出平均的训练度，平均的损失值和平均的精确度
    print("train_loss: " + str(loss/n))
    print("train_acc: " + str(current/n))

#定义验证函数，没有反向传播
def val(dataloader, model, loss_fn):
    #模型进行验证模式
    model.eval()
    loss, current, n = 0.0, 0.0, 0
    with torch.no_grad():#在模型不参与更新的情况下
        for batch, (X, y) in enumerate(dataloader):
            # 向前传播
            #X, y = X.to(device), y.to(device)
            output = model(X)
            cur_loss = loss_fn(output, y)  # 计算损失值
            _, pred = torch.max(output, axis=1)  # 输出每行最大的概率，意思是在10个概率中取最大的概率，然后取出概率对应最大的索引值
            cur_acc = torch.sum(y == pred) / output.shape[0]  # 计算精确度

            loss += cur_loss.item()
            current += cur_acc.item()
            n = n + 1
            # 输出平均的训练度，平均的损失值和平均的精确度
        print("val_loss: " + str(loss / n))
        print("val_acc: " + str(current / n))

        return current/n

#开始训练
epoch = 50#定义训练轮次
max_acc = 0.0

print(device)
for t in range(epoch):
    print(f'--------------------\nepoch{t+1}')
    #训练的
    train(train_dataloader, model, loss_fn, optimizer)
    #验证的
    a = val(test_dataloader, model, loss_fn)
    #保存最好的模型权重
    if a > max_acc:
        folder = 'save_model'
        if not os.path.exists(folder):
            os.mkdir('save_model')
        max_acc = a
        print('save best model')
        torch.save(model.state_dict(), 'save_model/best_model.pth')
print('Done!')
