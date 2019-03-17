import torch
import torch.optim as optim

import torchtext
from torchtext import data
import spacy

import argparse
import os


from models import *

spacy_en = spacy.load('en')

def tokenizer(text): # create a tokenizer function
    return [tok.text for tok in spacy_en.tokenizer(text)]

def load_model(embdim, vocab, n_filters):

        ######

        # 3.4 YOUR CODE HERE
        baseline_model = Baseline(embedding_dim=embdim, vocab=vocab)
        RNN_model = RNN(embedding_dim=embdim, vocab=vocab, hidden_dim=100)
        CNN_model = CNN(embedding_dim=embdim, vocab=vocab, n_filters=n_filters, filter_sizes=[2,4], kernel_size=100)

        loss_fnc = torch.nn.BCELoss()

        ######

        return baseline_model, RNN_model, CNN_model, loss_fnc

def evaluate(model, val_iter, loss_fnc):
    total_corr = 0

    for i, val_batch in enumerate(val_iter, 0):

        labels, feats = val_batch.Label, val_batch.Text[0]

        predictions = model(feats.transpose(0, 1), val_batch.Text[1], feats.shape[0])

        batch_loss = loss_fnc(input=abs(predictions.squeeze()), target=abs(labels.float().squeeze()))

        corr = (predictions > 0.5).squeeze().long() == labels

        total_corr += int(corr.sum())

    return float(total_corr) / len(val_iter.dataset), batch_loss

def testfnc(model, test_iter, loss_fnc):
    total_corr = 0

    for i, test_batch in enumerate(test_iter, 0):
        labels, feats = test_batch.Label, test_batch.Text[0]

        predictions = model(feats.transpose(0, 1), test_batch.Text[1], feats.shape[0])

        batch_loss = loss_fnc(input=abs(predictions.squeeze()), target=abs(labels.float().squeeze()))

        corr = (predictions > 0.5).squeeze().long() == labels

        total_corr += int(corr.sum())

    return float(total_corr) / len(test_iter.dataset), batch_loss

def main(args):
    ######

    # 3.2 Processing of the data

    TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, include_lengths=True)
    LABEL = data.Field(sequential=False, use_vocab=False)
    # instantiate data.Field objects

    train, val, test = data.TabularDataset.splits(
        path='./data/', train='train.tsv',
        validation='val.tsv', test='test.tsv', format='tsv',
        fields=[('Text', TEXT), ('Label', LABEL)])
    # load in train/val/test

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test), sort_key=lambda x: len(x.Text),
        batch_sizes=(64, 64, 64), device="cpu", sort_within_batch=True, repeat=False)
    # split into batches

    TEXT.build_vocab(train, vectors="glove.6B.100d")
    # load pre-trained word vectors

    vocab = TEXT.vocab

    ######

    ######

    # 5 Training and Evaluation

    print("Training start")

    step = 0  # tracks batch number

    baseline_model, RNN_model, CNN_model, loss_fnc= load_model(embdim=args.embdim, vocab=vocab, n_filters=args.numfilt)

    if args.model =='baseline':
        current_model = baseline_model
    elif args.model == 'rnn':
        current_model = RNN_model
    elif args.model == 'cnn':
        current_model = CNN_model

    optimizer = torch.optim.Adam(current_model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        total_train_loss = 0.0
        tot_corr = 0

        for i, batch in enumerate(train_iter, 0):
            labels, feats = batch.Label, batch.Text[0]

            step += 1

            # zero the gradients
            optimizer.zero_grad()

            predictions = current_model(feats.transpose(0,1), batch.Text[1], feats.shape[0])

            batch_loss = loss_fnc(input=abs(predictions.squeeze()), target=abs(labels.float().squeeze()))
            # compute current batch loss function based on label data

            total_train_loss += batch_loss
            # update total train loss

            batch_loss.backward()
            optimizer.step()
            # adjust parameters based on gradient

            corr = (predictions > 0.5).squeeze().long() == labels
            tot_corr += int(corr.sum())

            if (step + 1) % 10 == 0: # evaluate model on validation data every eval_every steps
                this_val_acc, this_val_loss = evaluate(current_model, val_iter, loss_fnc)
                print("Epoch: {}, Step {} | Training Loss: {} | Validation acc: {} | Validation loss: {}".format(epoch + 1, step + 1, float(total_train_loss) / (i + 1), this_val_acc, this_val_loss))

        print("train acc: ", float(tot_corr) / len(train_iter.dataset))

    this_test_acc, this_test_loss = testfnc(current_model, test_iter, loss_fnc)
    print("Test acc: {} | Test loss: {}".format(this_test_acc, this_test_loss))

    if args.model == 'baseline':
        torch.save(current_model, 'model_baseline_doryan.pt')
    elif args.model == 'rnn':
        torch.save(current_model, 'model_RNN_doryan.pt')
    elif args.model == 'cnn':
        torch.save(current_model, 'model_CNN_doryan.pt')

    ######
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--model', type=str, default='rnn',
                        help="Model type: baseline,rnn,cnn (Default: baseline)")
    parser.add_argument('--embdim', type=int, default=100)
    parser.add_argument('--rnn-hidden-dim', type=int, default=100)
    parser.add_argument('--numfilt', type=int, default=50)

    args = parser.parse_args()

    main(args)