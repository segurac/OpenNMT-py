from __future__ import division

import copy
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch import cuda
import torch.optim as optim
from torch.autograd import Variable

import onmt
import onmt.Models
import onmt.ModelConstructor
import onmt.modules
from onmt.Utils import aeq, use_gpu
from MyModel import MyRNN
import opts
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

parser = argparse.ArgumentParser(description='AnswerSelection.py')

# opts.py
opts.add_md_help_argument(parser)
opts.model_opts(parser)
opts.train_opts(parser)
opts.train_AS(parser)

opt = parser.parse_args()
torch.manual_seed(1369)

if opt.gpuid:
    cuda.set_device(opt.gpuid[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)

if len(opt.gpuid) > 1:
    sys.stderr.write("Sorry, multigpu isn't supported yet, coming soon!\n")
    sys.exit(1)


def make_train_data_iter(train_data, opt):
    """
    This returns user-defined train data iterator for the trainer
    to iterate over during each train epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    like curriculum learning is ok too.
    """
    # Sort batch by decreasing lengths of sentence required by pytorch.
    # sort=False means "Using dataset's sortkey instead of iterator's sortkey".
    return onmt.IO.OrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=False)


def make_valid_data_iter(valid_data, opt):
    """
    This returns user-defined validate data iterator for the trainer
    to iterate over during each validate epoch. We implement simple
    ordered iterator strategy here, but more sophisticated strategy
    is ok too.
    """
    return onmt.IO.OrderedIterator(
                dataset=valid_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                train=True, sort=True)


def make_pool_data_iter(train_data, opt):
    return onmt.IO.MyOrderedIterator(
                dataset=train_data, batch_size=opt.pool_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=True, train=True, sort=False, shuffle=True)


def load_fields(train, valid, checkpoint):
    fields = onmt.IO.ONMTDataset.load_fields(
                torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.IO.ONMTDataset.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab)))

    return fields


def collect_features(train, fields):
    # TODO: account for target features.
    # Also, why does fields need to have the structure it does?
    src_features = onmt.IO.ONMTDataset.collect_features(fields)
    aeq(len(src_features), train.nfeatures)

    return src_features


def my_trainer(train_iter, valid_iter, model, opt):

    criterion = nn.MarginRankingLoss(opt.margin)
    cos_dist = nn.CosineSimilarity()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # batchN_iter = iter(pool_iter)

    running_loss = 0.0
    for i, batch in enumerate(train_iter):
        target_size = batch.tgt.size(0)
        # batchN = next(batchN_iter)

        dec_state = None
        _, src_lengths = batch.src

        question = onmt.IO.make_features(batch, 'src')
        c_answer = onmt.IO.make_features(batch, 'tgt')
        n_answer = onmt.IO.make_features(batch, 'pool')

        model.zero_grad()
        q, a, a_n = model.forward(question, c_answer, n_answer, src_lengths)

        p_dist = cos_dist(q, a)
        n_dist = cos_dist(q, a_n)

        targets = torch.ones(batch.batch_size)
        if opt.gpuid:
            targets = targets.cuda()

        targets = Variable(targets)

        loss = criterion(p_dist, n_dist, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.data[0]

    print(i)
    return running_loss/i


def main():
    opt.batch_size = 64
    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)
    print('Example', train.examples[10000-1].src)

    checkpoint = None
    model_opt = opt

    # Load fields generated from preprocess phase.
    fields = load_fields(train, valid, checkpoint)
    opt.vocab_size = len(fields['src'].vocab)

    # Collect features.
    src_features = collect_features(train, fields)
    for j, feat in enumerate(src_features):
        print(' * src feature %d size = %d' % (j, len(fields[feat].vocab)))

    # Build Model
    model = MyRNN(opt.vocab_size, opt.word_vec_size,
                  opt.QA_rnn_size, opt.QALSTM, opt.dropout,
                  opt.batch_size, opt.QAbrnn, use_cuda=opt.gpuid)
    if opt.gpuid:
        model.cuda()

    # Data iterators
    train_iter = make_train_data_iter(train, opt)
    valid_iter = make_valid_data_iter(valid, opt)
    # pool_iter = make_pool_data_iter(train, opt)

    all_losses = []
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        running_loss = my_trainer(train_iter, valid_iter, model, opt)
        all_losses.append(running_loss)
        print('*' * 5 + 'Epoch ' + str(epoch) + '*' * 5 + '----> loss = ' + str(running_loss))

    plt.figure()
    plt.plot(all_losses)
    plt.savefig('loss-vs-time.png')


if __name__ == "__main__":
    main()
