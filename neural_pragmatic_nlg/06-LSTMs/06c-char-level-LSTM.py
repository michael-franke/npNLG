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
import urllib.request
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
# with urllib.request.urlopen("https://raw.githubusercontent.com/michael-franke/npNLG/main/neural_pragmatic_nlg/05-RNNs/names-data.json") as url:
#     namesData = json.load(url)

# with open('names-data.json') as dataFile:
#     namesData = json.load(dataFile)

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
    # print(k, total_size, train_size, test_size)
    indices       = [i for i in range(total_size)]
    random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices  = indices[(train_size+1):(-1)]
    train_data[k] = [namesData[k][i] for i in train_indices]
    test_data[k]  = [namesData[k][i] for i in test_indices]

##################################################
## define LSTM
##################################################

class LSTM(nn.Module):
    def __init__(self, cat_embedding_size, n_cat,
                 char_embedding_size, n_char,
                 hidden_size, output_size, num_layers = 2, dropout = 0.1):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # category embedding
        self.cat_embedding = nn.Embedding(n_cat, cat_embedding_size)
        # character embedding
        self.char_embedding = nn.Embedding(n_char, char_embedding_size)
        # the actual LSTM
        self.lstm = nn.LSTM(input_size  = cat_embedding_size+char_embedding_size,
                            hidden_size = hidden_size,
                            num_layers  = num_layers,
                            batch_first = True,
                            dropout = dropout
                            )
        # linear map onto weights for words
        self.linear_map = nn.Linear(hidden_size, output_size)

    def forward(self, category, name, hidden):
        cat_emb  = self.cat_embedding(category)
        char_emb = self.char_embedding(name)
        output, (hidden, cell) = self.lstm(torch.concat([cat_emb, char_emb], dim = 1))
        predictions = self.linear_map(output)
        return torch.nn.functional.log_softmax(predictions, dim = 1), hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

##################################################
## helper functions for training
##################################################

# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random name from that category
def randomTrainingPair():
    category = randomChoice(categories)
    name = randomChoice(train_data[category])
    return category, name

# get index representation of name (in the proper format)
def getNameIndices(name):
    indices = [SOSIndex] + [all_letters.index(c) for c in list(name)] + [EOSIndex]
    return indices

# get index representation of category (in the proper format)
# NB: must have same length as corresponding name representation b/c
#     each character in the sequence is concatenated with the category information
def getCatIndices(category, name_length):
    return torch.full((1,name_length), categories.index(category)).reshape(-1)

# get random training pair in desired input format (vectors of indices)
def randomTrainingExample():
    category, name = randomTrainingPair()
    name_length = len(name) + 2
    return getCatIndices(category, name_length), torch.tensor(getNameIndices(name))

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

##################################################
## single training pass
##################################################

def train(cat, name):
    # get a fresh hidden layer
    hidden = lstm.initHidden()
    # zero the gradients
    optimizer.zero_grad()
    # run sequence
    predictions, hidden = lstm(cat, name, hidden)
    # compute loss (NLLH)
    loss = criterion(predictions[:-1], name[1:len(name)])
    # perform backward pass
    loss.backward()
    # perform optimization
    optimizer.step()
    # return prediction and loss
    return loss.item()

##################################################
## actual training loop
## (should take about 1-2 minutes)
##################################################

# instantiate model
lstm = LSTM(cat_embedding_size  = 32,
            n_cat               = n_categories,
            char_embedding_size = 32,
            n_char              = n_letters,
            hidden_size         = 64,
            output_size         = n_letters,
            dropout             = 0,
            num_layers          = 1
            )
# training objective
criterion = nn.NLLLoss(reduction='sum')
# learning rate
learning_rate = 0.005
# optimizer
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
# training parameters
n_iters = 50000
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
    name      = torch.tensor(getNameIndices(name))
    cat       = getCatIndices(category,len(name))
    hidden    = lstm.initHidden()
    prediction, hidden = lstm(cat, name, hidden)
    nll       = criterion(prediction[:-1], name[1:len(name)])
    return(nll.item())

def get_surprisal_dataset(data):
    surprisl_dict = dict()
    surp_avg_dict = dict()
    perplxty_dict = dict()
    for category in list(data.keys()):
        surprisl = 0
        surp_avg = 0
        perplxty = 0
        # training
        for name in data[category]:
            item_surpr = get_surprisal_item(category, name)
            surprisl  += item_surpr
            surp_avg  += item_surpr / len(name)
            perplxty  += item_surpr ** (-1 / len(name))
        n_items = len(data[category])

        surprisl_dict[category] = (surprisl /n_items)
        surp_avg_dict[category] = (surp_avg / n_items)
        perplxty_dict[category] = (perplxty / n_items)

    return(surprisl_dict, surp_avg_dict, perplxty_dict)

def makeDF(surp_dict):
    p = pandas.DataFrame.from_dict(surp_dict)
    p = p.transpose()
    p.columns = ["surprisal", "surp_scaled", "perplexity"]
    return(p)

surprisal_test  = makeDF(get_surprisal_dataset(test_data))
surprisal_train = makeDF(get_surprisal_dataset(train_data))

print("\nmean surprisal (test):", np.mean(surprisal_test["surprisal"]))
print("\nmean surprisal (train):", np.mean(surprisal_train["surprisal"]))

##################################################
## prediction function
##################################################

max_length = 20

# make a prediction based on given sequence
def predict(category, initial_sequence, decode_strat = "greedy"):

    if len(initial_sequence) >= max_length:
        return(initial_sequence)

    name      = torch.tensor(getNameIndices(initial_sequence))[:-1]
    cat       = getCatIndices(category,len(name))
    hidden    = lstm.initHidden()

    generation = initial_sequence

    output, hidden = lstm(cat, name, hidden)
    next_word_pred = output[-1]

    if decode_strat == "pure":
        sample_index = torch.multinomial(input = torch.exp(next_word_pred),
                                         num_samples = 1)
        pass
    else:
        topv, topi = next_word_pred.topk(1)
        sample_index = topi[0].item()

    if sample_index == EOSIndex:
        return(generation)
    else:
        generation += all_letters[sample_index]

    return(predict(category, generation))

print(predict("German", "", decode_strat = "greedy"))
print(predict("German", "", decode_strat = "pure"))
print(predict("German", "", decode_strat = "pure"))
print(predict("German", "", decode_strat = "pure"))

print(predict("Japanese", "", decode_strat = "greedy"))
print(predict("Japanese", "", decode_strat = "pure"))
print(predict("Japanese", "", decode_strat = "pure"))
print(predict("Japanese", "", decode_strat = "pure"))

# extend the 'predict' function to include a parameter to implement the following decoding schemes:
# - top-k (variable k)
# - softmax (variable soft-max parameter)
# - top-p (variable p)

def infer_category(name):
    probs = torch.tensor([torch.exp(-torch.tensor(get_surprisal_item(c, name))) for c in categories])
    probs = probs/torch.sum(probs)
    vals, cats = probs.topk(3)
    print("Top 3 guesses for ", name, ":\n")
    for i in range(len(cats)):
        print("%12s: %.5f" %
              (categories[cats[i]], vals[i].detach().numpy() ))

# that's really interesting: there is a base-rate effect!
infer_category("Smith")
infer_category("Miller")
