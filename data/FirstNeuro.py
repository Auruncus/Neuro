#!/usr/bin/env python
# coding: utf-8

# In[9]:


import torch 
import torch.nn as nn 
import torchvision.datasets as dsets 
import torchvision.transforms as transforms 
from torch.autograd import Variable 
  
# Hyper Parameters  
input_size = 784
num_classes = 10
num_epochs = 5
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
class LogisticRegression(nn.Module): 
    def __init__(self, input_size, num_classes): 
        super(LogisticRegression, self).__init__() 
        self.linear = nn.Linear(input_size, num_classes) 
  
    def forward(self, x): 
        out = self.linear(x) 
        return out 
  
  
model = LogisticRegression(input_size, num_classes) 
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


# In[15]:


optimizer


# In[16]:


total


# In[17]:


torch.save(model.state_dict(), 'neuro_model.pkl')


# In[ ]:


torch.cuda.device_count()


# In[ ]:


labels[3]


# In[ ]:


from matplotlib import pyplot as plt 
import numpy as np 
first_image = images[3] 
first_image = np.array(first_image, dtype='float') 
pixels = first_image.reshape((28, 28)) 
plt.imshow(pixels, cmap='gray') 
plt.show() 


# In[ ]:


import torch 
print(torch.rand(3,3).cuda())


# In[ ]:


if torch.cuda.is_available():
    print("cuda is available")

