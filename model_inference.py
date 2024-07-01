import torch
import torch.nn as nn
from model import Net

def inference(device: nn.Module, model_path: str, data_loader: torch.utils.data.DataLoader, 
              training_epochs: list[int], criterion):
    avg_losses = []
    accuracies = []

    for training_epoch in training_epochs:
        ######## Load Model #########
        print(f"Load Model after training epoch {training_epoch}")

        net = Net()
        net.load_state_dict(torch.load(model_path + f'_epoch{training_epoch}.pt'))
        net.to(device)
        net.eval()

        ######## Inference #########
        total_loss = 0.0
        total = 0
        correct = 0

        for i, data in enumerate(data_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device) # inputs, labels = data

            # perform inference
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            #print(outputs)
            #print(predicted)
            #print(labels)
            #print(loss)

            # calculate statistics (loss and accuracy)
            total_loss += loss.item()
            total += 1
            correct += (predicted == labels).item()

            #print(total)
            #print(correct)

            if i % 1000 == 999:    # print progress every 1000 samples
                print(f'completed {i + 1} iterations')

        print(f"After epoch {training_epoch}, results: total_loss = {total_loss}, total = {total}, correct = {correct}")
        
        avg_losses.append(total_loss / len(data_loader.dataset))
        accuracies.append(correct / total * 100)
    
    return avg_losses, accuracies