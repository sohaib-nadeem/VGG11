import torch
import torch.nn as nn
import torch.optim as optim
from model import Net

def train(device: nn.Module, model_save_path: str, data_loader: torch.utils.data.DataLoader,
          epochs: int, learning_rate: float, momentum: float, criterion):
    ######## Create Model and Optimizer #########
    net = Net()
    print(net)
    params = list(net.parameters())
    print(len(params))
    print(params[0].size())  # conv1's .weight
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    ######## Begin Training #########
    for epoch in range(1, epochs + 1):  # loop over the dataset multiple times
        print(f"Begin training for epoch {epoch}")

        running_loss = 0.0
        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            #inputs, labels = data
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
            
        ######## Save Model At Every Epoch #########
        print(f"Save Model after training epoch {epoch}")
        torch.save(net.state_dict(), model_save_path + f"_epoch{epoch}.pt")

    print(f'Finished Training for {epochs} Epochs')