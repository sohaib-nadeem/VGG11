import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # covolutional layers
        self.conv1 = nn.Conv2d(1, 64, 3, 1, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv5 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv6 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv7 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv8 = nn.Conv2d(512, 512, 3, 1, 1)      

        # fully connected linear layers
        self.fc1 = nn.Linear(512, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

        # dropout 
        self.drop1 = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)

        # BatchNorm
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.batchNorm2 = nn.BatchNorm2d(128)
        self.batchNorm3 = nn.BatchNorm2d(256)
        self.batchNorm4 = nn.BatchNorm2d(256)
        self.batchNorm5 = nn.BatchNorm2d(512)
        self.batchNorm6 = nn.BatchNorm2d(512)
        self.batchNorm7 = nn.BatchNorm2d(512)
        self.batchNorm8 = nn.BatchNorm2d(512)

    def forward(self, input):
        ################################################################################

        # Conv(001, 064, 3, 1, 1) - BatchNorm(064) - ReLU - MaxPool(2, 2)
        c1 = F.relu(self.batchNorm1(self.conv1(input)))
        s2 = F.max_pool2d(c1, (2, 2), 2)

        # Conv(064, 128, 3, 1, 1) - BatchNorm(128) - ReLU - MaxPool(2, 2)
        c3 = F.relu(self.batchNorm2(self.conv2(s2)))
        s4 = F.max_pool2d(c3, (2, 2), 2)

        # Conv(128, 256, 3, 1, 1) - BatchNorm(256) - ReLU
        c5 = F.relu(self.batchNorm3(self.conv3(s4)))

        # Conv(256, 256, 3, 1, 1) - BatchNorm(256) - ReLU - MaxPool(2, 2)
        c6 = F.relu(self.batchNorm4(self.conv4(c5)))
        s7 = F.max_pool2d(c6, (2, 2), 2)

        # Conv(256, 512, 3, 1, 1) - BatchNorm(512) - ReLU
        c8 = F.relu(self.batchNorm5(self.conv5(s7)))

        # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
        c9 = F.relu(self.batchNorm6(self.conv6(c8)))
        s10 = F.max_pool2d(c9, (2, 2), 2)

        # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU
        c11 = F.relu(self.batchNorm7(self.conv7(s10)))

        # Conv(512, 512, 3, 1, 1) - BatchNorm(512) - ReLU - MaxPool(2, 2)
        c12 = F.relu(self.batchNorm8(self.conv8(c11)))
        s13 = F.max_pool2d(c12, (2, 2), 2)

        ################################################################################

        # Flatten operation: purely functional, outputs a (N, ?) Tensor
        s13 = torch.flatten(s13, 1)

        # FC(0512, 4096) - ReLU - Dropout(0.5)
        f14 = self.drop1(F.relu(self.fc1(s13)))

        # FC(4096, 4096) - ReLU - Dropout(0.5)
        f15 = self.drop2(F.relu(self.fc2(f14)))

        # FC(4096, 10)
        output = self.fc3(f15)
        
        return output