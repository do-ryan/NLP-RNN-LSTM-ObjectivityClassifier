"""
    6. Testing on Your Own Sentence.

    You will write a python script that prompts the user for a sentence input on the command line, and prints the
    classification from each of the three models, as well as the probability that this sentence is subjective.

    An example console output:

        Enter a sentence
        What once seemed creepy now just seems campy

        Model baseline: subjective (0.964)
        Model rnn: subjective (0.999)
        Model cnn: subjective (1.000)

        Enter a sentence
"""

import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os

import numpy as np

from models import *

spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def main():

    baseline_model = torch.load('model_baseline_doryan.pt')
    rnn_model = torch.load('model_RNN_doryan.pt')
    cnn_model = torch.load('model_CNN_doryan.pt')

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)

    train, val, test = data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    TEXT.build_vocab(train, vectors="glove.6B.100d")
    # load pre-trained word vectors

    vocab = TEXT.vocab

    while True:
        print("Enter a sentence ('exit' to quit): ")
        sentence = input()

        if sentence == "exit":
            break

        tokenized_sentence = tokenizer(sentence)

        enumerated_sentence = []

        for token in tokenized_sentence:
            enumerated_sentence.append(vocab.stoi[token])
        # convert tokenized sentence to numeric form


        tokentensor = torch.LongTensor(enumerated_sentence).view(1,-1)
        sentence_length = [tokentensor.shape[1]]

        probability = baseline_model(tokentensor)
        print("Model baseline: {} {}".format(probability > 0.5 and 'subjective' or 'objective', probability.detach().numpy().squeeze()))
        probability = rnn_model(tokentensor, sentence_length)
        print("Model rnn: {} {}".format(probability > 0.5 and 'subjective' or 'objective', probability.detach().numpy().squeeze()))
        probability = cnn_model(tokentensor, sentence_length, tokentensor.shape[1])
        print("Model cnn: {} {}".format(probability > 0.5 and 'subjective' or 'objective', probability.detach().numpy().squeeze()))

    return

if __name__ == '__main__':

    main()
