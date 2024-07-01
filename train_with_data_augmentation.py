import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from model import Net
from model_train import train

############################### Select Device ###########################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

############################### Load Data ###########################################
print("#################### Load Data #########################")

# to augment the model, we will add three transforms:
# - a horizontal flip with probabilty 0.5
# - a vertical flip with probabilty 0.5
# - gaussian noise with standard deviation x, 
#   where x is sampled from a uniform[0, 1) distribution
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.RandomHorizontalFlip(p=0.5),
     transforms.RandomVerticalFlip(p=0.5),
     transforms.Lambda(lambda x : x + torch.rand(1).item() * torch.randn_like(x))
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

training_set_proportion = 1 # limit size of training set to speed up training, unused
batch_size = 4
download = True

mnist_training_data = torchvision.datasets.MNIST('./data', train=True,
                                        download=download, transform=transform)
trainloader = torch.utils.data.DataLoader(mnist_training_data,
                                          batch_size=batch_size,
                                          shuffle=True, num_workers=0)


############################# Training ###################################################
print("#################### Training #########################")

epochs = 5
criterion = nn.CrossEntropyLoss()
# parameters for SGD optimizer:
learning_rate = 0.001 
momentum = 0.9

train(device, './augmented_model', trainloader, epochs, learning_rate, momentum, criterion)
