import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose,transforms
from PIL import Image


l_function = nn.CrossEntropyLoss()
accuracy_counter =[]
loss_counter = []
image =  Image.open('img_2.jpg')
transformImage = transforms.Compose([transforms.ToTensor()])
image_tensor = transformImage(image)
class network(nn.Module):
    def __init__(self):
        super(network,self).__init__()
        self.net = nn.Sequential(
            nn.Linear(784,512),
            nn.ReLU(),
            nn.Linear(512,10)
            nn.LogSoftmax(dim=1)
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

def model_train(data,model,optim,loss_function):
    print("Starting training------------------------")
    size=  len(data.dataset)
    model.train()
    model_loss=[]
    
    for current_batch, (train, expected) in enumerate(data):
        predicted_value = model(train)
        loss_value = loss_function(predicted_value,expected)
        model_loss.append(loss_value.detach())
        opti.zero_grad()
        loss_value.backward()
        opti.step()
    
    loss_counter.append(torch.tensor(model_loss).mean())
def model_test(data, model):
    data_size = len(data.dataset)
    model.eval()
    with torch.no_grad():
        x, target =  next(iter(data))
        logits = model(x)
        acc = accuracy(logits, target)
        accuracy_counter.append(acc)

transform =  transforms.Compose([transforms.ToTensor(),flatten])
mnist_train_data =  datasets.MNIST(root="./", train = True,download=False,transform=transform)
mnist_test_data =  datasets.MNIST(root="./",train = False,download=False,transform=transform) 
train_data = data.DataLoader(mnist_train_data,batch_size=64,shuffle=True)
test_data = data.DataLoader(mnist_test_data,batch_size=64,shuffle=True)
image =  data.DataLoader(image_tensor)
# for i in range(20):
#     model_train(train_data,model,opti,l_function)
#     model_test(test_data,model)
print(model(image_tensor))
# print(accuracy_counter)
# print(loss_counter)