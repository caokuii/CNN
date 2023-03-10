#lenet5_test
import torch
from net import MyLeNet5
from torch.autograd import Variable
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage

#数据转化为tensor格式
data_transform = transforms.Compose([
    transforms.ToTensor()
])

#加载训练数据集
train_dataset = datasets.MNIST(root='./data', train= True, transform= data_transform, download= True)
train_dataloader = torch.utils.data.DataLoader(dataset= train_dataset, batch_size= 16, shuffle= True)

#加载测试集数据，在这里肯定只用到了测试数据
test_dataset = datasets.MNIST(root='./data', train= False, transform= data_transform, download= True)
test_dataloader = torch.utils.data.DataLoader(dataset= test_dataset, batch_size= 16, shuffle= True)

#如果有显卡，可以转到GPU，因为我的电脑没有独立显卡所以选择直接在CPU上训练
device = "cuda" if torch.cuda.is_available() else 'cpu'
#device = 'cpu'

#调用net里面定义的模型，将模型数据转到device中
#model = MyLeNet5()
model =  MyLeNet5().to(device)

#加载已经训练好的最好的模型
model.load_state_dict(torch.load("D:/AStudy/python_study/CNN/LeNet5/save_model/best_model.pth"))

#获取结果，方便输出预测的值和真实的值
classes = [
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
]

#把tensor转化为图片，方便可视化
show = ToPILImage()

#进入验证
#取前5张图片
for i in range(5):
    #test_dataset的每一项的第一个数据是张量，第二个数据是标签
    X, y = test_dataset[i][0], test_dataset[i][1]
    show(X).show()

    X = Variable(torch.unsqueeze(X, dim= 0).float(), requires_grad = False).to(device)
    with torch.no_grad():
        pred = model(X)
        #取出预测值和真实值
        predicted, actual = classes[torch.argmax(pred[0])], classes[y]

        print(f'predicted: "{predicted}", actual: "{actual}')





