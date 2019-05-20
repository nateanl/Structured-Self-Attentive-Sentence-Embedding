import torch
import torch.nn as nn
from torch.autograd import Variable

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class BiLSTM(nn.Module):

    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.drop = nn.Dropout(config['dropout'])
        self.word_embedding = nn.Embedding(config['word_vocab'], config['word_size'])
        self.pos_embedding = nn.Embedding(config['pos_vocab'], config['pos_size'])
        self.word_embedding.from_pretrained(config['word_matrix'])
        self.pos_embedding.from_pretrained(config['pos_matrix'])
        self.word_embedding.weight.requires_grad = False
        self.pos_embedding.weight.requires_grad = False
        self.word_bilstm = nn.LSTM(config['ninp_word'], config['word_nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True, batch_first=True)
        self.pos_bilstm = nn.LSTM(config['ninp_pos'], config['pos_nhid'], config['nlayers'], dropout=config['dropout'],
                              bidirectional=True, batch_first=True)
        self.nlayers = config['nlayers']
        self.nhid_word = config['word_nhid']
        self.nhid_pos = config['pos_nhid']
        self.pooling = config['pooling']
        self.dictionary = config['dictionary']

    def forward(self, word_ids, pos_ids, hidden):
        word_hidden,pos_hidden = hidden
        word_emb = self.drop(self.word_embedding(word_ids))
        pos_emb = self.drop(self.pos_embedding(pos_ids))
        pos_outp = self.pos_bilstm(pos_emb, pos_hidden)[0]
        word_outp = self.word_bilstm(word_emb, word_hidden)[0]
        return word_outp, pos_outp, word_emb, pos_emb

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return ((Variable(weight.new(self.nlayers * 2, bsz, self.nhid_word).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid_word).zero_())),
                (Variable(weight.new(self.nlayers * 2, bsz, self.nhid_pos).zero_()),
                Variable(weight.new(self.nlayers * 2, bsz, self.nhid_pos).zero_())))

class SelfAttentiveEncoder(nn.Module):

    def __init__(self, config):
        super(SelfAttentiveEncoder, self).__init__()
        self.bilstm = BiLSTM(config)
        self.drop = nn.Dropout(config['dropout'])
        self.pos_ws1 = nn.Linear(config['pos_nhid'] * 2, config['attention-unit'], bias=False)
        self.word_ws1 = nn.Linear(config['word_nhid'] * 2, config['attention-unit'], bias=False)
        self.pos_ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.word_ws2 = nn.Linear(config['attention-unit'], config['attention-hops'], bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.dictionary = config['dictionary']
#        self.init_weights()
        self.attention_hops = config['attention-hops']

    def init_weights(self, init_range=0.1):
        self.pos_ws1.weight.data.uniform_(-init_range, init_range)
        self.pos_ws2.weight.data.uniform_(-init_range, init_range)
        self.word_ws1.weight.data.uniform_(-init_range, init_range)
        self.word_ws2.weight.data.uniform_(-init_range, init_range)

    def get_word_attention(self, outp, inp):
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.word_ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.word_ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def get_pos_attention(self, outp, inp):
        size = outp.size()  # [bsz, len, nhid]
        compressed_embeddings = outp.contiguous().view(-1, size[2])  # [bsz*len, nhid*2]
        transformed_inp = inp.view(size[0], 1, size[1])  # [bsz, 1, len]
        concatenated_inp = [transformed_inp for i in range(self.attention_hops)]
        concatenated_inp = torch.cat(concatenated_inp, 1)  # [bsz, hop, len]

        hbar = self.tanh(self.pos_ws1(self.drop(compressed_embeddings)))  # [bsz*len, attention-unit]
        alphas = self.pos_ws2(hbar).view(size[0], size[1], -1)  # [bsz, len, hop]
        alphas = torch.transpose(alphas, 1, 2).contiguous()  # [bsz, hop, len]
        penalized_alphas = alphas + (
            -10000 * (concatenated_inp == self.dictionary['<pad>']).float())
            # [bsz, hop, len] + [bsz, hop, len]
        alphas = self.softmax(penalized_alphas.view(-1, size[1]))  # [bsz*hop, len]
        alphas = alphas.view(size[0], self.attention_hops, size[1])  # [bsz, hop, len]
        return torch.bmm(alphas, outp), alphas

    def forward(self, word_ids, pos_ids, hidden):
        word_outp, pos_outp, word_emb, pos_emb = self.bilstm.forward(word_ids, pos_ids, hidden)
        word_o, word_a = self.get_word_attention(word_outp, word_ids)
        pos_o, pos_a = self.get_pos_attention(pos_outp, pos_ids)
        return word_o,pos_o, word_a, pos_a

    def init_hidden(self, bsz):
        return self.bilstm.init_hidden(bsz)

class Classifier(nn.Module):

    def __init__(self, config):
        super(Classifier, self).__init__()
        if config['pooling'] == 'mean' or config['pooling'] == 'max':
            self.encoder = BiLSTM(config)
            self.fc = nn.Linear((config['word_nhid']+config['pos_nhid']) * 2, config['nfc'])
        elif config['pooling'] == 'all':
            self.encoder = SelfAttentiveEncoder(config)
            self.fc = nn.Linear((config['word_nhid']+config['pos_nhid']) * 2 * config['attention-hops'], config['nfc'])
        else:
            raise Exception('Error when initializing Classifier')
        self.drop = nn.Dropout(config['dropout'])
        self.tanh = nn.Tanh()
        self.pred = nn.Linear(config['nfc'], config['class-number'])
        self.dictionary = config['dictionary']
#        self.init_weights()

    def init_weights(self, init_range=0.1):
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.pred.weight.data.uniform_(-init_range, init_range)
        self.pred.bias.data.fill_(0)

    def forward(self, word_id, pos_id, hidden):
        word_o,pos_o, word_a, pos_a = self.encoder.forward(word_id,pos_id, hidden)
        outp = torch.cat((word_o, pos_o),dim=2)
        outp = outp.view(outp.size(0), -1)
        fc = self.tanh(self.fc(self.drop(outp)))
        pred = self.pred(self.drop(fc))
        if type(self.encoder) == BiLSTM:
            attention = None
        return pred, word_a, pos_a

    def init_hidden(self, bsz):
        return self.encoder.init_hidden(bsz)

    def encode(self, inp, hidden):
        return self.encoder.forward(inp, hidden)[0]
