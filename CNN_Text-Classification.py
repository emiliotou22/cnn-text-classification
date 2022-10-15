#!/usr/bin/env python
# coding: utf-8

# ## Applications of NLP:
# - Sentiment Analysis
# - Chatbot
# - Speech Recognition
# - Machine Translation
# - Advertisement Matching
# - Text Classification

# ## Types of Deep Learning Models:
# 1. Artificial Neural Networks (ANN) - one input layer, multiple hidden layers and one output layer
# 2. Convolutional Neural Networks (CNN) - focuses on feature; convolution and pooling layers (feature extraction), fully connected layer and output (classification)
# 3. Recurrent Neural Networks (RNN) - similar structure to ANN; passes data forward and backwards

# In[2]:


pip install torch


# In[3]:


import torch
import numpy as np


# In[7]:


# Construct a tensor from an array
array = [[1,2], [7,4],[5,6]]
tensor0 = torch.tensor(array)
print(tensor0)
print('The data structure type of tensor0:', type(tensor0))
print('The data type of tensor0:', tensor0.dtype)
print('The sdape of tensor0:', tensor0.shape)


# In[10]:


# Construct a tensor from a nupmy array
np_array = np.array([[1,2], [7,4],[5,6]])
tensor1 = torch.tensor(np_array)
print(tensor0)
print('The data structure type of tensor0:', type(tensor0))
print('The data type of tensor0:', tensor0.dtype)
print('The sdape of tensor0:', tensor0.shape)


# #### Slicing

# In[11]:


tensorA = torch.tensor([[1,1,1], [2,2,2]])
tensorB = torch.tensor([[3,3,3],[4,4,4,]])


# In[12]:


# Slicing is all teh same as numpy arrays
print('Slicing the first two rows of tensorA(index one inclusive index two exclusive)')
print(tensorA[:2])
print('Slicing the frist two colimuns of tensorA (take all rows, tehn slice columns)')
print(tensorA[:, :2])


# ##### Concatenation

# In[14]:


print('Vertically concatenate tensorA and tensorB (dim=0)')
concat_v = torch.cat([tensorA, tensorB])
print(concat_v)


# In[15]:


print('Horizontally concatenate tensorA and tensorB (dim=1)')
concat_h = torch.cat([tensorA, tensorB], dim=1)
print(concat_h)


# ### Using CNN

# In[15]:


pip install torchtext==0.10.0


# In[16]:


import torch
import torchtext
from torchtext.legacy import data, datasets
import random


# In[8]:


seed = 966
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[9]:


# define fields
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField()


# In[ ]:


train, test = dadtasets.TREC.splits(TEXT, LABEL)
train, val = train.split(random_state= random.seed(seed))


# In[ ]:


vars(train[-1])


# In[ ]:


# build vocab
TEXT.build_vocab(train, min_freq=2)
LABEL.build_vocab(train)


# In[ ]:


print(LABEL.vocab.stoi)


# In[ ]:


print('Voacab size of TEXT', len(TEXT.vocab.stoi))
print('Voacab size of LABEL', len(LABEL.vocab.stoi))


# In[ ]:


train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
(train, val, test),
batch_size = 64,
sort_key = lambda x: len(x.text),
device = device
)


# #### Building CNN

# In[17]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[ ]:


class CNN(nn.Module):
    def __init__(self, vocabulary_size, embedding_size,
                kernels_number, kernel_sizes, output_size, dropout_rate):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.convolution_layers = nn.ModuleList([nn.Conv2d(in_channels=1,
                                                          out_channels=kernels_number,
                                                          kernel_size=(k, embedding_size))
                                                for k in kernel_sizes])
        
        self.dropout = nn.Dropout(dropout_rate)
        self.fully_connected = nn.Linear(len(kernel_sizes) * kernels_number, output_size)
        
    def forward(self, text):
        text = text.permute(1,0)
        input_embeddings = self.embeddings(text)
        input_embeddings = input_embeddings.unsqueeze(1)
        conved = [F.relu(convolution_layer(input_embeddings)).squeeze(3)
                 for convolution_layer in self.convolution_layers]
        pooled = [F.max_polld(conv, conv.shape[2]).squeeze(2) for conv in conved]
        concat = self.dropout(torch.cat(pooled, dim=1))
        final_output = self.fully_connected(concat)
        
        return final_output


# In[ ]:


vocabulary_size = 2679
embedding_size = 100
kernels_number = 100
kernels_sizes = [2,3,4]
output_size = 6


# In[ ]:


model = CNN(vocabulary_size, embedding_size,
                kernels_number, kernel_sizes, output_size, dropout_rate)


# In[ ]:


print(model)


# In[18]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


# In[ ]:


import torch.optim as optim
import torch.nn as nn

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

optimizer = optim.Adam(model.parameters())


# In[ ]:


def accuracy(predictions, actual_label):
    max_predictions = predictions.argmax(dim=1, keepdim=True, )
    correct_predictions = max_predictions.squeeze(1).eq(actual_label)
    accuracy = correct_predictions.sum() / torch.cuda.FloatTensor([actual_label.shape[0]])
    return accuracy


# In[ ]:


def train(model, iterator, optimizer, criterion):
    
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in iterator:
        # optimizer
        optimizer.zero_grad()
        # predictions
        predictions = model(batch.text)
        # loss
        loss = criterion(predictions, batch.label)
        # accuracy
        acc = accuracy(predictions, batch.label)
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# In[ ]:


def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    with torch.no_grad():
        
        for batch in iterator:
            
            predictions = model(batch.text)
            
            loss = criterion(predictions, batch.label)
            
            acc = categorical_accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


# #### Training the model

# In[ ]:


number_of_epochs = 20

best_acc = float('-inf')

for epoch in range(number_of_epochs);
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    
    valid_loss, valid_acc = evaluate(model, train_iterator, criterion)
    
    if valid_acc > best_acc:
        best_acc = valid_acc
        torch.save(model.state_dict(), 'trec.pt')
    
    print(f'Epoch {epoch+1} ')
    print(f'\tTraing Loss: {train_loss: .3f} | Train Acc: {train_acc*100: .2f}%')
    print(f'\t Validation Loss: {valid_loss: .3f} | Validation Acc: {valid_acc*100: .2f}%')
    


# In[ ]:


model.load_state_dict(torch.load('trec.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss: .3f} | Test Acc: {test_acc*100: .2f}%')


# In[ ]:




