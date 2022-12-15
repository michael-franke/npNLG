import torch
import torch.nn as nn
# import time
# import math

# starting point: raw text input
input_raw1 = "Colin was here."
input_raw2 = "Colin went away subito."

# vocab definition: all units that we want to model
# include special tokens:
# like "[UNK]" for unknown words (here: "subito")
vocabulary = ["Colin", "was", "here", "went", "away", "[UNK]" "[EOS]", "[PAD]"]
n_vocab = len(vocabulary) + 1

# tokenization: sequence of tokens from the vocabulary
input_tok1 = ["Colin", "was", "here", "[EOS]", "[PAD]"]
input_tok2 = ["Colin", "went", "away", "subito", "[EOS]"]

# indexing: represent input as tensor of indices
input_ind1 = torch.tensor([0,1,2,6,7])
input_ind2 = torch.tensor([0,3,4,5,6])

batch_size = 2
sequence_length = 5

batched_input = torch.cat([input_ind1, input_ind2]).reshape(batch_size, sequence_length)
print(batched_input)

# embedding: vector-representation of each word
# needs parameter for the size of the embedding
n_embbeding_size = 3
embedding = nn.Embedding(n_vocab, n_embbeding_size)
# embedding is a matrix of size (n_vocab, n_embedding_size)
for p in embedding.parameters():
    print(p)

# the 'embedding' takes batched input
input_embedding = embedding(batched_input)
print(input_embedding)

n_hidden_size = 4

# instantiate an LSTM
lstm = nn.LSTM(input_size  = n_embbeding_size,
               hidden_size = n_hidden_size,  # size of hidden state
               num_layers  = 2,              # number of stacked layers
               batch_first = True            # expect input in format (batch, sequence, token)
               )

# apply LSTM function (initial hidden state and initial cell state default to all 0s)
# LSTM maps a (batched) sequence of embeddings, onto a triple:
#   outcome: sequence of hidden states in final layer for each word
#   hidden:  hidden states of last word for each layer
#   cell:    cell states of last word for each layer
output, (hidden, cell) = lstm(input = input_embedding)

print("\nLSTM embeddings in last layer for each word:\n", output)
print("\nHidden state of last word for each layer:\n", hidden)
print("\nCell state of last word for each layer:\n", cell)

linear_map = nn.Linear(n_hidden_size, n_vocab)
weights = linear_map(output)

# get probabilities from output weights
def next_word_probabilities(weights):
    softmax = torch.nn.Softmax(dim=2)
    return(softmax(weights).detach().numpy().round(4))

# without any training: next word weights have high entropy
print(next_word_probabilities(weights))
