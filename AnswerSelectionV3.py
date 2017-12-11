from __future__ import division

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
from MyModel import MyRNN, MyRNN_CNN, MyAttetion_RNN_CNN, MyAttentionRNN
import opts
import numpy as np
import time

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

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
                device=opt.gpuid[0] if opt.gpuid else -1, repeat=False,
                train=False, sort=True)


def make_pool_data_iter(train_data, opt):
    return onmt.IO.MyOrderedIterator(
                dataset=train_data, batch_size=opt.batch_size,
                device=opt.gpuid[0] if opt.gpuid else -1,
                repeat=True, train=False, sort=False, shuffle=True)


def load_fields(train, valid, checkpoint):
    fields = onmt.IO.MyONMTDataset.load_fields(
                torch.load(opt.data + '.vocab.pt'))
    fields = dict([(k, f) for (k, f) in fields.items()
                  if k in train.examples[0].__dict__])
    train.fields = fields
    valid.fields = fields

    if opt.train_from:
        print('Loading vocab from checkpoint at %s.' % opt.train_from)
        fields = onmt.IO.MyONMTDataset.load_fields(checkpoint['vocab'])

    print(' * vocabulary size. source = %d; target = %d; pool = %d' %
          (len(fields['src'].vocab), len(fields['tgt'].vocab), len(fields['pool'].vocab)))

    return fields


def collect_features(train, fields):
    # TODO: account for target features.
    # Also, why does fields need to have the structure it does?
    src_features = onmt.IO.MyONMTDataset.collect_features(fields)
    aeq(len(src_features), train.nfeatures)

    return src_features


def my_trainer(train_iter, model, opt, epoch):

    criterion = nn.MarginRankingLoss(opt.margin)
    cos_dist = nn.CosineSimilarity()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model.train()
    targets = torch.ones(opt.batch_size)
    previous = opt.batch_size
    if opt.gpuid:
        targets = targets.cuda()
        targets = Variable(targets)

    running_loss = 0.0
    for batch_idx, batch in enumerate(train_iter):

        if batch_idx > opt.fraction_db*len(train_iter):
            break

        _, src_lengths = batch.src

        question = onmt.IO.make_features(batch, 'src')
        c_answer = onmt.IO.make_features(batch, 'tgt')
        n_answer = onmt.IO.make_features(batch, 'pool')

        if question.size(0) == 1:
            continue

        model.zero_grad()
        q, a, a_n = model.forward(question, c_answer, n_answer, src_lengths, batch.batch_size, 0)

        p_dist = cos_dist(q, a)
        n_dist = cos_dist(q, a_n)

        if batch.batch_size != previous:
            previous = batch.batch_size
            targets.data.resize_(q.size(0))
            targets.data.fill_(1)

        loss = criterion(p_dist, n_dist, targets)
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.data[0]

        if (batch_idx+1) % 200 == 0:
            print('\r Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * batch.batch_size, len(train_iter.dataset),
                       100. * batch_idx / len(train_iter), loss.data[0]))

    return running_loss/batch_idx


def my_validator(valid_iter, pool_iter, model, opt):
    criterion = nn.MarginRankingLoss(opt.margin)
    cos_dist = nn.CosineSimilarity()
    valid_loss = 0
    correct = 0

    model.eval()
    targets = torch.ones(opt.batch_size)
    equals = torch.zeros(opt.batch_size).long()
    if opt.gpuid:
        targets = targets.cuda()
        targets = Variable(targets, volatile=True)
        equals = equals.cuda()

    list_q = []
    list_a = []

    get_out = False
    num_batch = 0
    num_samples = 0
    for batch_idx, batch in enumerate(valid_iter):

        if batch_idx > opt.fraction_db*len(valid_iter):
            get_out = True

        _, src_lengths = batch.src

        question = onmt.IO.make_features(batch, 'src')
        c_answer = onmt.IO.make_features(batch, 'tgt')

        if question.size(0) > 1:
            q, a = model.forward(question, c_answer, None, src_lengths, batch.batch_size, 1)

            list_q.append(q)
            list_a.append(a)
            num_batch += 1
            num_samples += batch.batch_size

        if batch_idx % 200 == 0:
            print('\rProcessing validation... {} / {}'.format(batch_idx,  len(valid_iter)))

        # Every n batches compute validation
        if (batch_idx + 1) % opt.n_batch_valid != 0 and batch_idx != (len(valid_iter)-1) and get_out is False:
            continue

        print("Computing validation...")
        all_q = torch.cat((list_q[0], list_q[1]), 0)
        for i in range(2, len(list_q)):
            all_q = torch.cat((all_q, list_q[i]), 0)

        rnd_idx = np.random.randint(0, all_q.size(0),
                                    size=(opt.batch_size, opt.pool_size, len(list_q)))
        rnd_idx_t = torch.from_numpy(rnd_idx).cuda()

        for i in range(0, len(list_q)):
            q = list_q[i]
            a = list_a[i]

            p_dist = cos_dist(q, a)

            list_n_a = []
            for j in range(0, opt.pool_size):
                n_a = make_random_batch(all_q, rnd_idx_t[:q.size(0), j, i])
                n_dist = cos_dist(q, n_a)
                list_n_a.append(n_dist)

            targets.data.resize_(q.size(0))
            targets.data.fill_(1)
            equals.resize_(q.size(0))
            equals.fill_(0)

            aux_loss = 0
            for neg in list_n_a:
                aux_loss += criterion(p_dist, neg, targets).data[0]
            aux_loss /= opt.pool_size
            valid_loss += aux_loss

            p_dist = p_dist.view(q.size(0), 1)
            all_answers = torch.cat((p_dist, list_n_a[0].view(q.size(0), 1)), 1)
            for k in range(1, len(list_n_a)):
                n_dist = list_n_a[k]
                n_dist = n_dist.view(q.size(0), 1)
                all_answers = torch.cat((all_answers, n_dist), 1)

            aux = all_answers.data
            _, pred = torch.max(aux, 1)
            correct += pred.eq(equals).sum()

        # Free memory
        list_q = []
        list_a = []
        list_n_a = []
        del all_answers

        if get_out is True:
            break

    return valid_loss/num_batch, correct/num_samples


def make_random_batch(all_q, rnd_idx):
    n_a = torch.index_select(all_q, 0, rnd_idx)
    return n_a


def main():
    opt.batch_size = 20
    db_name = os.path.split(os.path.dirname(opt.data))[-1]

    # Load train and validate data.
    print("Loading train and validate data from '%s'" % opt.data)
    train = torch.load(opt.data + '.train.pt')
    valid = torch.load(opt.data + '.valid.pt')
    print(' * number of training sentences: %d' % len(train))
    print(' * maximum batch size: %d' % opt.batch_size)
    # print('Example', train.examples[10000-1].src)

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
    if opt.network == 'RNN':
        model = MyRNN(opt.vocab_size, opt.word_vec_size,
                      opt.QA_rnn_size, opt.layers_QALSTM, opt.dropout,
                      opt.batch_size, opt.QAbrnn, use_cuda=opt.gpuid)
    elif opt.network == 'CNN':
        model = MyRNN_CNN(opt.vocab_size, opt.word_vec_size,
                          opt.QA_rnn_size, opt.n_filters, opt.window_size, opt.layers_QALSTM, opt.dropout,
                          opt.batch_size, opt.QAbrnn, use_cuda=opt.gpuid)
    elif opt.network == 'Attention_RNN':
        model = MyAttentionRNN(opt.vocab_size, opt.word_vec_size,
                               opt.QA_rnn_size, opt.layers_QALSTM,
                               opt.attention_size, opt.dropout, opt.batch_size, opt.QAbrnn, use_cuda=opt.gpuid)
    elif opt.network == 'Attention_CNN':
        model = MyAttetion_RNN_CNN(opt.vocab_size, opt.word_vec_size,
                                   opt.QA_rnn_size, opt.n_filters, opt.window_size, opt.layers_QALSTM,
                                   opt.attention_size, opt.dropout, opt.batch_size, opt.QAbrnn, use_cuda=opt.gpuid)

    print(model)
    if opt.gpuid:
        model.cuda()

    # Data iterators
    train_iter = make_train_data_iter(train, opt)
    valid_iter = make_valid_data_iter(valid, opt)
    pool_iter = make_pool_data_iter(train, opt)

    train_losses = []
    val_losses = []
    val_accuracy = []

    print(26*"*"), print(5*"*" + " Start training " + 5*"*"), print(26*"*")
    result_file = open('models/' + db_name + '/results_' + opt.name + '.txt', 'a')

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train_loss = my_trainer(train_iter, model, opt, epoch)
        train_losses.append(train_loss)
        val_loss, val_acc = my_validator(valid_iter, pool_iter, model, opt)
        val_losses.append(val_loss)
        val_accuracy.append(val_acc)

        print('\n' + '*' * 8 + 'Epoch ' + str(epoch) + '*' * 8 + '\n')
        print('\r Train set: Average loss: {:.4f}'.format(train_loss))
        print('\r Valid set: Average loss: {:.4f}, Accuracy: {:.0f}%\n'.format(
            val_loss, 100.*val_acc))
        print('\n' + '*' * 22 + '\n')

        plt.figure()
        plt.plot(train_losses, label='train')
        plt.plot(val_losses, label='valid')
        plt.legend()
        plt.savefig('models/' + db_name + '/' + opt.name + '_loss-vs-epoch.png')
        plt.close()

        plt.figure()
        plt.plot(val_accuracy, label='valid')
        plt.savefig('models/' + db_name + '/' + opt.name + '_Accuracy-vs-epoch.png')
        plt.close()

        torch.save(model, 'models/' + db_name + '/' + opt.name + '_' + str(epoch) + '_acc_' + str(val_acc) + '.pt')
        result_file.write("{} \r {} \r {}\n".format(train_loss, val_loss, val_acc))
        result_file.flush()

    result_file.close()


if __name__ == "__main__":
    main()
