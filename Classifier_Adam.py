import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose,transforms

def accuracy(logits, target):
    predicted = logits.argmax(dim=1)
    return (predicted == target).type(torch.float).mean()
def flatten(data):
    return data.reshape(-1)
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,10),
            nn.Softmax(dim=1)
        )
    def forward(self,input):
        input = torch.flatten(input,1)
        output = self.net(input)
        return output
model =  network()
print(model)
