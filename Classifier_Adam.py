import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose,transforms

l_function = nn.CrossEntropyLoss()
accuracy_counter =[]
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
model = network()
print(model)
opti = optim.Adam(model.parameters())
def accuracy(logits, target):
    predicted = logits.argmax(dim=1)
    return (predicted == target).type(torch.float).mean()
def flatten(data):
    return data.reshape(-1)

def model_train(data,model, optim, loss_function):
    model.train()
    for current_batch, (train, expected) in enumerate(data):
        predicted_value = model(train)
        loss_value = loss_function(predicted_value,expected)

        opti.zero_grad()
        loss_value.backward()
        opti.step()

def model_test(data, model):
    data_size = len(data.dataset)
    model.eval()
    x, target =  next(iter(data))
    logits = model(x)
    acc = accuracy(logits, target)
    accuracy_counter.append(acc)

transform =  transforms.Compose([transforms.ToTensor(),flatten])
mnist_train_data =  datasets.MNIST(root="./", train = True,download=False,transform=transform)
mnist_test_data =  datasets.MNIST(root="./",train = False,download=False,transform=transform) 
train_data = data.DataLoader(mnist_train_data,batch_size=64)
test_data = data.DataLoader(mnist_test_data,batch_size=64)

for i in range(5):
    model_train(train_data,model,opti,l_function)
    model_test(test_data,model)
print(accuracy_counter)