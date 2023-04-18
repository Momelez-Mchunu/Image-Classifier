import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose,transforms
from PIL import Image
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

accuracy_counter =[]
loss_counter = []

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


def accuracy(logits, target):
    predicted = logits.argmax(dim=1)
    return (predicted == target).type(torch.float).mean()
def flatten(data):
    return data.reshape(-1)

def model_train(data,model,optim,loss_function):
    size=  len(data.dataset)
    model.train()
    model_loss=[]
    
    for current_batch, (train, expected) in enumerate(data):
        predicted_value = model(train)
        loss_value = loss_function(predicted_value,expected)
        model_loss.append(loss_value.detach())
        optim.zero_grad()
        loss_value.backward()
        optim.step()
        if current_batch%100==0:
            current = current_batch*len(train)
            percentage_done =  round(current/size,2)*100
            print(f"{percentage_done}% done")
    
    loss_counter.append(torch.tensor(model_loss).mean())
def model_test(data, model):
    data_size = len(data.dataset)
    model.eval()
    with torch.no_grad():
        x, target =  next(iter(data))
        logits = model(x)
        acc = accuracy(logits, target)
        accuracy_counter.append(acc)
def obtain_trained():
    model = network()
    print(model)
    opti = optim.Adam(model.parameters())
    l_function = nn.CrossEntropyLoss()
    transform =  transforms.Compose([transforms.ToTensor(),flatten])
    mnist_train_data =  datasets.MNIST(root="./", train = True,download=True,transform=transform)
    mnist_test_data =  datasets.MNIST(root="./",train = False,download=True,transform=transform) 
    train_data = data.DataLoader(mnist_train_data,batch_size=64,shuffle=True)
    test_data = data.DataLoader(mnist_test_data,batch_size=64,shuffle=True)
    for i in range(10):
        print(f"--------------------Iteration {i+1}--------------------")

        model_train(train_data,model,opti,l_function)
        model_test(test_data,model)
    torch.save(model,'models/Classifier_Adam.pt')

def plot(accuracy,loss):
    plt.plot(accuracy,label='Validation Accuraccy')
    plt.plot(loss,label='Training Loss')
    plt.legend(frameon=False)
    plt.savefig('plot.png')
    plt.show()
def predict_image(image):
    saved_model = torch.load('models/Classifier_Adam.pt')
    saved_model.eval()
    image_in =  Image.open(image)
    transformImage = transforms.Compose([transforms.ToTensor()])
    image_tensor = transformImage(image_in)
    output = saved_model(image_tensor)
    index =  output.data.cpu().numpy().argmax()
    return index

# print(predict_image("MNIST_JPGS/trainingSample/img_3.jpg"))
def main():
    print(sys.argv[0])
    if sys.argv[0] =="Classifier_Adam.py":
        if os.path.exists("models/Classifier_Adam.pt"):
            user_input = input("Please enter a filepath:\n> ")
            while (user_input!='exit'):
                value =  predict_image(user_input)
                print(f"Classifier: {value}")
                user_input = input("Please enter a filepath:\n> ")
        else:
            obtain_trained()
            print("Done!")
            user_input = input("Please enter a filepath:\n> ")
            while (user_input!='exit'):
                value =  predict_image(user_input)
                print(f"Classifier: {value}")
                user_input = input("Please enter a filepath:\n> ")

if __name__ == "__main__":
    main()
