#!/usr/bin/env python
# coding: utf-8

# In[50]:


import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
import torch.nn.functional as fun
from torch.autograd import Variable 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyper Parameters  
input_size = 784
num_classes = 10
hidden_size = 300  
num_epochs = 4
batch_size = 100
learning_rate = 0.1
# MNIST Dataset (Images and Labels) 
train_dataset = dsets.MNIST(root ='./data', 
							train = True, 
							transform = transforms.ToTensor(), 
							download = True) 

test_dataset = dsets.MNIST(root ='./data', 
						train = False, 
						transform = transforms.ToTensor()) 

# Dataset Loader (Input Pipline) 
train_loader = torch.utils.data.DataLoader(dataset = train_dataset, 
										batch_size = batch_size, 
										shuffle = True) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, 
										batch_size = batch_size, 
										shuffle = False) 

# Model 
class Perceptron(nn.Module): 
    def __init__(self, input_size, hidden_size, num_classes): 
        super(Perceptron, self).__init__()  
        self.pc1 = nn.Linear(28*28, 300)
        self.pc2 = nn.Linear(300, 300)
        self.pc3 = nn.Linear(300, 10)
        self.relu = nn.ReLU()
  
    def forward(self, x): 
        out=self.relu(self.pc1(x))
        out=self.relu(self.pc2(out))
        out = self.pc3(out)
        #out = fun.relu(self.pc1(x))
        #out = fun.relu(self.pc2(out))
        #out = self.pc3(out)
        return out 
  
  
model = Perceptron(input_size, hidden_size, num_classes) 
#model.cuda()
  
# Loss and Optimizer 
# Softmax is internally computed. 
# Set parameters to be updated. 
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate) 
  
# Training the Model

for epoch in range(num_epochs): 
    for i, (images, labels) in enumerate(train_loader): 
        images = Variable(images.view(-1, 28 * 28)) 
        labels = Variable(labels) 
  
        # Forward + Backward + Optimize 
        optimizer.zero_grad() 
        outputs = model(images)
        loss = criterion(outputs, labels) 
        loss.backward() 
        optimizer.step() 
  
        if (i + 1) % 100 == 0: 
            print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, 
                     len(train_dataset) // batch_size, loss.item())) 
  
# Test the Model 
correct = 0
total = 0
for images, labels in test_loader: 
    images = Variable(images.view(-1, 28 * 28)) 
    outputs = model(images) 
    _, predicted = torch.max(outputs.data, 1) 
    total += labels.size(0) 
    correct += (predicted == labels).sum() 
  
print('Accuracy of the model on the 10000 test images: % d %%' % ( 
            100 * correct / total)) 


# In[49]:


from matplotlib import pyplot as plt 
import numpy as np 
for k in range(0,10):
    first_image = images[k] 
    first_image = np.array(first_image, dtype='float') 
    pixels = first_image.reshape((28, 28)) 
    plt.imshow(pixels, cmap='gray') 
    print("Предсказание:",predicted[k])
    print("Действительно:",labels[k])
    plt.show() 

