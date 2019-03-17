import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0) # set seed for torch RNG


class Baseline(nn.Module):
    def __init__(self, embedding_dim, vocab):
        super(Baseline, self).__init__()

        ######

        # 4.1 YOUR CODE HERE

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.fc1 = nn.Linear(in_features=vocab.vectors.shape[1], out_features=1)

        ######

        return
    def forward(self, x, lengths=None, max_sentence_length=None):

        ######

        # 4.1 YOUR CODE HERE

        x = self.embedding(x) #(batch size x num tokens x word vector size)
        x = torch.mean(x, 1) #(batch size x word vector size)
        x = self.fc1(x) #(batch_size x prediction)
        x = torch.sigmoid(x)
        return x

        ######

class RNN(nn.Module):
    def __init__(self, embedding_dim, vocab, hidden_dim):
        super(RNN, self).__init__()

        ######

        # 4.2 YOUR CODE HERE

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.GRU = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim) # should be 100 and 100 respectively
        self.fc1 = nn.Linear(in_features=embedding_dim, out_features=1)

        ######
        return

    def forward(self, x, lengths, max_sentence_length=None):

        ######

        # 4.2 YOUR CODE HERE

        x = self.embedding(x)
        x = nn.utils.rnn.pack_padded_sequence(x, lengths=lengths, batch_first=True)
        #output, h_n = self.GRU(torch.transpose(x,1,0)) # when omitting pack_padded_sequence
        output, h_n = self.GRU(x)
        x = self.fc1(h_n)
        x = torch.sigmoid(x)
        return(x)

        ######


class CNN(nn.Module):
    def __init__(self, embedding_dim, vocab, n_filters, filter_sizes, kernel_size):
        super(CNN, self).__init__()

        ######

        # 4.3 YOUR CODE HERE

        #assume hidden dimension = 100, embedding_dim = 100

        self.embedding = nn.Embedding.from_pretrained(vocab.vectors)
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[0], embedding_dim))
        self.conv1b = nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(filter_sizes[1], embedding_dim))
        self.pool1a = nn.MaxPool1d(kernel_size=kernel_size-1, stride=1)
        self.pool1b = nn.MaxPool1d(kernel_size=kernel_size-3, stride=1)
        self.fc1 = nn.Linear(in_features= 2*n_filters, out_features=1)

        return


    def forward(self, x, lengths, max_sentence_length):
        ######

        # 4.3 YOUR CODE HERE

        self.pool1a = nn.MaxPool1d(kernel_size=max_sentence_length - 1, stride=1)
        self.pool1b = nn.MaxPool1d(kernel_size=max_sentence_length - 3, stride=1)
        # update maxpool kernel size with every use of model (token sequences length for different batches are different)

        x = self.embedding(x).unsqueeze(1) # add an extra dimension to index 1 for the 4d kernel to work
        y = self.pool1a(torch.relu(torch.squeeze(self.conv1a(x), 3)))
        z = self.pool1b(torch.relu(torch.squeeze(self.conv1b(x), 3)))
        x = torch.cat((y,z), dim=1)
        x = x.view(-1, 100)
        x = self.fc1(x)
        x = torch.sigmoid(x)
        return x

