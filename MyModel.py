import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack


class MyRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_layers, dropout, batch_size, bidirectional, use_cuda):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                encoder state
        outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        super(MyRNN, self).__init__()
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_direct = 2 if bidirectional else 1

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

    def forward(self, question, c_answers, n_answers, lengths, batch_size, isvalid):
        lengths = lengths.view(-1).tolist()

        # Process Question
        hidden = self.init_hidden(batch_size)
        q_size = question.size()
        question = question.view(q_size[0], -1)
        q_emb = self.embeddings(question)
        s_len, batch, emb_dim = q_emb.size()
        q_packed_emb = pack(q_emb, lengths)

        q_outputs, q_hidden_t = self.lstm(q_packed_emb, hidden)
        q_outputs = unpack(q_outputs)[0]
        q_outputs, _ = torch.max(q_outputs, 0)  # MaxPooling

        # Process correct Answer
        hidden = self.init_hidden(batch_size)
        a_size = c_answers.size()
        c_answers = c_answers.view(a_size[0], -1)
        a_emb = self.embeddings(c_answers)
        s_len, batch, emb_dim = a_emb.size()
        # a_packed_emb = pack(a_emb, lengths)

        a_outputs, a_hidden_t = self.lstm(a_emb, hidden)
        # a_outputs = unpack(a_outputs)[0]
        a_outputs, _ = torch.max(a_outputs, 0)

        # Process negative answer
        if not isvalid:
            n_size = n_answers.size()
            hidden = self.init_hidden(batch_size)
            n_answers = n_answers.view(n_size[0], -1)
            n_emb = self.embeddings(n_answers)
            s_len, batch, emb_dim = n_emb.size()
            # n_packed_emb = pack(n_emb, lengths)

            n_outputs, n_hidden_t = self.lstm(n_emb, hidden)
            # n_outputs = unpack(n_a_outputs)[0]
            n_outputs, _ = torch.max(n_outputs, 0)

            return q_outputs, a_outputs, n_outputs
        else:
            return q_outputs, a_outputs

    def init_hidden(self, batch_size):
        # h0 = Variable(torch.zeros(self.n_layers*self.n_direct, self.batch_size, self.hidden_size))
        # c0 = Variable(torch.zeros(self.n_layers*self.n_direct, self.batch_size, self.hidden_size))

        h0 = torch.zeros(self.n_layers*self.n_direct, batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers*self.n_direct, batch_size, self.hidden_size)

        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        h0 = Variable(h0)
        c0 = Variable(c0)

        return h0, c0


class MyRNN_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_filters, window_size,
                 n_layers, dropout, batch_size, bidirectional, use_cuda):
        """
        Args:
            input (LongTensor): len x batch x nfeat.
            lengths (LongTensor): batch
            hidden: Initial hidden state.
        Returns:
            hidden_t (Variable): Pair of layers x batch x rnn_size - final
                                encoder state
        outputs (FloatTensor):  len x batch x rnn_size -  Memory bank
        """
        super(MyRNN_CNN, self).__init__()
        self.use_cuda = use_cuda
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.n_filters = n_filters
        self.window_size = window_size
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.n_direct = 2 if bidirectional else 1

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=n_layers,
                            dropout=dropout,
                            bidirectional=bidirectional)

        self.conv1 = nn.Conv2d(1, n_filters, (hidden_size*self.n_direct, window_size))

    def forward(self, question, c_answers, n_answers, lengths, batch_size, isvalid):
        lengths = lengths.view(-1).tolist()

        # Process Question
        hidden = self.init_hidden(batch_size)
        q_size = question.size()
        question = question.view(q_size[0], -1)
        q_emb = self.embeddings(question)
        s_len, batch, emb_dim = q_emb.size()
        q_packed_emb = pack(q_emb, lengths)

        q_outputs, q_hidden_t = self.lstm(q_packed_emb, hidden)
        q_outputs = unpack(q_outputs)[0]
        q_outputs = torch.transpose(q_outputs, 0, 1)
        q_outputs = torch.transpose(q_outputs, 1, 2)
        q_outputs = torch.unsqueeze(q_outputs, 1)

        cnn_out = F.tanh(self.conv1(q_outputs))
        q_outputs = F.max_pool2d(cnn_out, (1, cnn_out.size(3)))
        q_outputs = torch.squeeze(q_outputs)

        # Process correct Answer
        hidden = self.init_hidden(batch_size)
        a_size = c_answers.size()
        c_answers = c_answers.view(a_size[0], -1)
        a_emb = self.embeddings(c_answers)
        s_len, batch, emb_dim = a_emb.size()
        # a_packed_emb = pack(a_emb, lengths)

        a_outputs, a_hidden_t = self.lstm(a_emb, hidden)
        a_outputs = torch.transpose(a_outputs, 0, 1)
        a_outputs = torch.transpose(a_outputs, 1, 2)
        a_outputs = torch.unsqueeze(a_outputs, 1)

        a_outputs = F.tanh(self.conv1(a_outputs))
        a_outputs = F.max_pool2d(a_outputs, (1, a_outputs.size(3)))
        a_outputs = torch.squeeze(a_outputs)

        # Process negative answer
        if not isvalid:
            n_size = n_answers.size()
            hidden = self.init_hidden(batch_size)
            n_answers = n_answers.view(n_size[0], -1)
            n_emb = self.embeddings(n_answers)
            s_len, batch, emb_dim = n_emb.size()
            # n_packed_emb = pack(n_emb, lengths)

            n_outputs, n_hidden_t = self.lstm(n_emb, hidden)
            n_outputs = torch.transpose(n_outputs, 0, 1)
            n_outputs = torch.transpose(n_outputs, 1, 2)
            n_outputs = torch.unsqueeze(n_outputs, 1)

            n_outputs = F.tanh(self.conv1(n_outputs))
            n_outputs = F.max_pool2d(n_outputs, (1, n_outputs.size(3)))
            n_outputs = torch.squeeze(n_outputs)

            return q_outputs, a_outputs, n_outputs
        else:
            return q_outputs, a_outputs

    def init_hidden(self, batch_size):
        # h0 = Variable(torch.zeros(self.n_layers*self.n_direct, self.batch_size, self.hidden_size))
        # c0 = Variable(torch.zeros(self.n_layers*self.n_direct, self.batch_size, self.hidden_size))

        h0 = torch.zeros(self.n_layers*self.n_direct, batch_size, self.hidden_size)
        c0 = torch.zeros(self.n_layers*self.n_direct, batch_size, self.hidden_size)

        if self.use_cuda:
            h0 = h0.cuda()
            c0 = c0.cuda()

        h0 = Variable(h0)
        c0 = Variable(c0)

        return h0, c0
