import json
import torch
import os
import time
import random
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from feature_generator import Yelp_Dataset
from model_word import Classifier, BiLSTM, SelfAttentiveEncoder, AverageMeter




word_matrix = torch.load('/scratch/near/anlp/word2vec_matrix.pt')
word_vocab_size, word2vec_dim = word_matrix.shape
f = open('/scratch/near/anlp/word_dictionary.json')
line = f.readlines()
word_dictionary = json.loads(line[0])
config = {}
config['dropout'] = 0.5
config['word_vocab'] = word_vocab_size
config['word_size'] = word2vec_dim
config['ninp'] = word2vec_dim
config['nhid']= 300
config['nlayers']=2
config['attention-unit']=350
config['attention-hops']=4
config['nfc'] = 512
config['class-number']=5
config['word_matrix'] = word_matrix
config['pooling'] = 'all'
config['dictionary'] = word_dictionary
config['penalization_coeff'] = 1.0

def Frobenius(mat):
    size = mat.size()
    if len(size) == 3:  # batched matrix
        ret = (torch.sum(torch.sum((mat ** 2), dim=1), 1).squeeze() + 1e-10) ** 0.5
        return torch.sum(ret) / size[0]
    else:
        raise Exception('matrix for computing Frobenius norm should be with 3 dims')

def train(model, train_loader, optimizer, epoch):
    losses = AverageMeter()
    times = AverageMeter()
    losses.reset()
    times.reset()
    model.train()
    for i, (word_id, pos_id, label) in enumerate(train_loader):
        begin = time.time()
        hidden = model.init_hidden(BATCH_SIZE)
        pred, attention = model(word_id, hidden)
        loss = criterion(pred, label)
        I = Variable(torch.zeros(BATCH_SIZE, config['attention-hops'], config['attention-hops']))
        for p in range(BATCH_SIZE):
            for q in range(config['attention-hops']):
                I.data[p][q][q] = 1
        I = I.to(cuda)
        attentionT = torch.transpose(attention, 1, 2).contiguous()
        extra_loss = Frobenius(torch.bmm(attention, attentionT)-I)
        loss += config['penalization_coeff'] * extra_loss
        losses.update(loss.item())
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        end = time.time()
        times.update(end-begin)
        print('epoch %d, %d/%d, training loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len(train_loader),losses.avg, times.avg*len(train_loader)), end='\r')
    print("\n")

def valid(model, valid_loader, optimizer, epoch):
    losses = AverageMeter()
    times = AverageMeter()
    losses.reset()
    times.reset()
    model.eval()
    with torch.no_grad():
        for i, (word_id, pos_id, label) in enumerate(valid_loader):
            begin = time.time()
            hidden = model.init_hidden(BATCH_SIZE)
            pred, attention = model(word_id, hidden)
            loss = criterion(pred, label)
            losses.update(loss.item())
            end = time.time()
            times.update(end-begin)
            print('epoch %d, %d/%d, validation loss: %f, time estimated: %.2f seconds'%(epoch, i+1,len(valid_loader),losses.avg, times.avg*len(valid_loader)), end='\r')
        print("\n")
    return losses.avg


BATCH_SIZE = 50
cuda = torch.device('cuda:1')
model = Classifier(config)
init_range = 0.1
model.encoder.bilstm.word_embedding.weight.data.uniform_(-init_range, init_range)
model.encoder.bilstm.word_embedding.weight[0,:]=0.0
model.encoder.bilstm.word_embedding.weight[1,:]=0.0
model.encoder.bilstm.word_embedding.weight.requires_grad = True
model.to(cuda)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, betas=[0.9, 0.999], eps=1e-8, weight_decay=0)
train_loader = DataLoader(Yelp_Dataset('train',cuda),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
valid_loader = DataLoader(Yelp_Dataset('dev',cuda),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
min_loss = float('inf')
count = 0
for epoch in range(0,50):
    train(model, train_loader, optimizer, epoch)
    valid_loss = valid(model, valid_loader, optimizer, epoch)
    if valid_loss<min_loss:
        count = 0
        min_loss = valid_loss
        model_path = '/scratch/near/anlp/saved_model_word_with_embedding/2/epoch_%d_%.2fmodel'%(epoch,valid_loss)
        directory = os.path.dirname(model_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(model,model_path)

    else:
        count+=1
        if count == 6:
            break
