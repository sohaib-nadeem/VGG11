import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from model import Net
from model_inference import inference

############################### Select Device ###########################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

############################### Load Data ###########################################
print("#################### Load Data #########################")

download = False

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((32, 32))
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

mnist_training_data = torchvision.datasets.MNIST('./data', train=True,
                                        download=download, transform=transform)
trainloader = torch.utils.data.DataLoader(mnist_training_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

mnist_test_data = torchvision.datasets.MNIST('./data', train=False,
                                        download=download, transform=transform)
testloader = torch.utils.data.DataLoader(mnist_test_data,
                                          batch_size=1,
                                          shuffle=True, num_workers=0)

############################# Shared Parameters for Inference #############################
training_epochs = [1, 2, 3, 4, 5]
criterion = nn.CrossEntropyLoss()

############################# Inference on Test Set #############################
print("#################### Inference on Test Set #########################")

test_losses, test_accuracies = inference(device, './model', testloader, 
                                         training_epochs, criterion)

# plot test accuracy vs the number of epochs
plt.scatter(training_epochs, test_accuracies)
plt.plot(training_epochs, test_accuracies)
plt.title('Test Accuracy vs Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Test Accuracy')
plt.savefig('test_accuracy_vs_epochs.png')
plt.clf()

# plot test loss vs the number of epochs
plt.scatter(training_epochs, test_losses)
plt.plot(training_epochs, test_losses)
plt.title('Test Loss vs Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Test Loss')
plt.savefig('test_loss_vs_epochs.png')
plt.clf()


############################# Inference on Training Set #############################
print("#################### Inference on Training Set #########################")

training_losses, training_accuracies = inference(device, './model', trainloader, 
                                                 training_epochs, criterion)

# plot training accuracy vs the number of epochs
plt.scatter(training_epochs, training_accuracies)
plt.plot(training_epochs, training_accuracies)
plt.title('Training Accuracy vs Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.savefig('training_accuracy_vs_epochs.png')
plt.clf()

# plot training loss vs the number of epochs
plt.scatter(training_epochs, training_losses)
plt.plot(training_epochs, training_losses)
plt.title('Training Loss vs Number of Epochs')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Loss')
plt.savefig('training_loss_vs_epochs.png')
plt.clf()