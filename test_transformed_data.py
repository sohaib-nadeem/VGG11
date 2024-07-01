import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

import math
from model import Net
from model_inference import inference

############################### Select Device ###########################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

############################# Shared Parameters for Inference #############################
training_epochs = [5]
criterion = nn.CrossEntropyLoss()
download = False

############################### Test Flip Types ###########################################
print("#################### Test Flip Types #########################")

transformNormal = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformNormal)
testloaderNormal = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderNormal, 
                               training_epochs, criterion)
baselineAccuracy = test_accuracies[0]


transformHorizontalFlip = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.RandomHorizontalFlip(p=1)
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformHorizontalFlip)
testloaderHorizontalFlip = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderHorizontalFlip, 
                               training_epochs, criterion)
horizontalFlipAccuracy = test_accuracies[0]


transformVerticalFlip = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.RandomVerticalFlip(p=1)
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformVerticalFlip)
testloaderVerticalFlip = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderVerticalFlip, 
                               training_epochs, criterion)
verticalFlipAccuracy = test_accuracies[0]


# plot test accuracy vs type of flip
plt.bar(["No Flip", "Horizontal Flip", "Vertical Flip"],
        [baselineAccuracy, horizontalFlipAccuracy, verticalFlipAccuracy])
plt.title('Test Accuracy for Different Flip Types')
plt.xlabel('Type of Flip')
plt.ylabel('Test Accuracy')
plt.savefig('test_accuracy_vs_flip_type.png')
plt.clf()


############################### Test Gaussian Noise ###########################################
print("#################### Test Gaussian Noise #########################")

transformGaussianNoise1 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.Lambda(lambda x : x + math.sqrt(0.01)*torch.randn_like(x))
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformGaussianNoise1)
testloaderGaussianNoise1 = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderGaussianNoise1, 
                               training_epochs, criterion)
gaussianNoise1Accuracy = test_accuracies[0]


transformGaussianNoise2 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.Lambda(lambda x : x + math.sqrt(0.1)*torch.randn_like(x))
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformGaussianNoise2)
testloaderGaussianNoise2 = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderGaussianNoise2, 
                               training_epochs, criterion)
gaussianNoise2Accuracy = test_accuracies[0]


transformGaussianNoise3 = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32)),
     transforms.Lambda(lambda x : x + math.sqrt(1)*torch.randn_like(x))
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transformGaussianNoise3)
testloaderGaussianNoise3 = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

_, test_accuracies = inference(device, './model', testloaderGaussianNoise3, 
                               training_epochs, criterion)
gaussianNoise3Accuracy = test_accuracies[0]


# plot test accuracy vs amount of gaussian noise
plt.bar(["No Noise", "Var = 0.01", "Var = 0.1", "Var = 1"],
        [baselineAccuracy, gaussianNoise1Accuracy, gaussianNoise2Accuracy, gaussianNoise3Accuracy])
plt.title('Test Accuracy for Different Amounts of Gaussian Noise Added')
plt.xlabel('Variance of Gaussian Noise')
plt.ylabel('Test Accuracy')
plt.savefig('test_accuracy_vs_gaussian_noise.png')
plt.clf()
