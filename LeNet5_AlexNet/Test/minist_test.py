import torch
from torchvision import datasets, transforms

data_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root= './data', train= True, transform= data_transforms, download= True)
#张量以及数字标签
image, label = train_dataset[0]

print(image)
print(label)