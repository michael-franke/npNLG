##################################################
## import packages
##################################################

from __future__ import unicode_literals, print_function, division
from io import open
import json
import glob
import os
import unicodedata
import pandas
import string
import torch
import numpy as np
import torch.nn as nn
import random
import time
import math
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

##################################################
## read and inspect the data
##################################################

os.chdir("/Users/micha/Library/Mobile Documents/com~apple~CloudDocs/xxx_sync_xxx/Work/Teaching/2022b-npNLG/book-repo/neural_pragmatic_nlg/05-NLMs/")

with open('names-data.json') as dataFile:
    namesData = json.load(dataFile)

categories = list(namesData.keys())
n_categories   = len(categories)

# we use all ASCII letters as the vocabulary (plus tokens [EOS], [SOS])
all_letters = string.ascii_letters + " .,;'-"
n_letters   = len(all_letters) + 2 # all letter plus [EOS] and [SOS] token
SOSIndex    = n_letters - 1
EOSIndex    = n_letters - 2

##################################################
## make a train/test split
##################################################

train_data = dict()
test_data  = dict()
split_percentage = 10
for k in list(namesData.keys()):
    total_size    = len(namesData[k])
    test_size     = round(total_size/split_percentage)
    train_size    = total_size - test_size
    print(k, total_size, train_size, test_size)
    indices       = [i for i in range(total_size)]
    random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices  = indices[(train_size+1):(-1)]
    train_data[k] = [namesData[k][i] for i in train_indices]
    test_data[k]  = [namesData[k][i] for i in test_indices]

##################################################
## define RNN
##################################################

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout = 0.1):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(n_categories + input_size + hidden_size,
                             hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size,
                             output_size)
        self.o2o = nn.Linear(hidden_size + output_size,
                             output_size)
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

##################################################
## helper functions for training
##################################################

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(categories)
    line = randomChoice(train_data[category])
    return category, line

# One-hot vector for category
def categoryTensor(category):
    li = categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including [EOS]) for input
# The first input is always [SOS]
def inputTensor(line):
    tensor = torch.zeros(len(line)+1, 1, n_letters)
    tensor[0][0][SOSIndex] = 1
    for li in range(len(line)):
        letter = line[li]
        tensor[li+1][0][all_letters.find(letter)] = 1
    return tensor

def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(len(line))]
    letter_indexes.append(EOSIndex)
    return torch.LongTensor(letter_indexes)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

##################################################
## single training pass
##################################################

def train(category_tensor, input_line_tensor, target_line_tensor):
    # reshape target tensor
    target_line_tensor.unsqueeze_(-1)
    # get a fresh hidden layer
    hidden = rnn.initHidden()
    # reset cumulative loss
    optimizer.zero_grad()
    loss = 0
    # zero the gradients
    # sequentially probe predictions and collect loss
    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        l = criterion(output, target_line_tensor[i])
        loss += l
    # perform backward pass
    loss.backward()
    # perform optimization
    optimizer.step()
    # return prediction and loss
    return loss.item() # / input_line_tensor.size(0)

##################################################
## actual training loop
## (should take about 2-4 minutes)
##################################################

# instantiate the model
rnn = RNN(n_letters, 128, n_letters)
# training objective
criterion = nn.NLLLoss()
# learning rate
learning_rate = 0.0005
# optimizer
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
# training parameters
n_iters = 100000
print_every = 5000
plot_every = 500
all_losses = []
total_loss = 0 # will be reset every 'plot_every' iterations

start = time.time()

for iter in range(1, n_iters + 1):
    loss = train(*randomTrainingExample())
    total_loss += loss

    if iter % plot_every == 0:
        all_losses.append(total_loss / plot_every)
        total_loss = 0

    if iter % print_every == 0:
        rolling_mean = np.mean(all_losses[iter - print_every*(iter//print_every):])
        print('%s (%d %d%%) %.4f' % (timeSince(start),
                                     iter,
                                     iter / n_iters * 100,
                                     rolling_mean))

##################################################
## monitoring loss function during training
##################################################

plt.figure()
plt.plot(all_losses)
plt.show()

##################################################
## evaluation
##################################################

def get_surprisal_item(category, name):
    category_tensor    = categoryTensor(category)
    input_line_tensor  = inputTensor(name)
    target_line_tensor = targetTensor(name)
    hidden             = rnn.initHidden()
    surprisal          = 0
    target_line_tensor.unsqueeze_(-1)

    for i in range(input_line_tensor.size(0)):
        output, hidden = rnn(category_tensor, input_line_tensor[i], hidden)
        surprisal += criterion(output, target_line_tensor[i])
    return(surprisal.item())

print(get_surprisal_item("German", "Franke"))
print(get_surprisal_item("Arabic", "Franke"))
print(get_surprisal_item("English", "Smith"))
print(get_surprisal_item("English", "SmXfs"))
