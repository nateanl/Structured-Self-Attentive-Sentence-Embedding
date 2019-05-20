import json
import spacy
import numpy as np
import pickle
from random import sample
from nltk import pos_tag
from nltk.corpus import masc_tagged
import word2vec
import nltk
import torch
# nltk.download('masc_tagged')
# nltk.download('averaged_perceptron_tagger')

def get_word_dictionary_matrix():
    embeddings_index = dict()
    f = open('/scratch/near/glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    f = open('./train.json')
    train_data = json.load(f)
    train_text = [ele['text'][:500] for ele in train_data]
    l = []
    for ele in train_text:
        l+=[a.lower() for a in ele]
    token_set = set(l)
    word_dictionary = {}
    word_matrix = []
    count = 2
    for ele in token_set:
        if ele in embeddings_index:
            word_dictionary[ele] = count
            count +=1
            word_matrix.append(embeddings_index[ele])
    word_dictionary['UNK'] = 1
    word_matrix.append(np.zeros(100,))
    word_dictionary['<pad>'] = 0
    word_matrix.append(np.zeros(100,))
    return word_dictionary, torch.tensor(word_matrix)

def get_pos_dictionary_matrix():
    txt_fname = 'tags.txt'
    vec_fname = 'vec.bin'
    vec_size = 15
    with open(txt_fname, 'w') as tags_file:
        words = masc_tagged.tagged_words()
        tags_file.write(' '.join([w[1] for w in words if w[1]]))

    word2vec.word2vec(
        txt_fname, vec_fname, size=vec_size,
        negative=5, sample=1, cbow=1, window=3, verbose=False)
    model = word2vec.load(vec_fname)
    pos_dictionary = {}
    count = 2
    for tag in model.vocab:
        pos_dictionary[tag] = count
        count +=1
    pos_dictionary['UNK'] = 1
    pos_dictionary['<pad>'] = 0
    pos_matrix = np.concatenate((np.zeros((2,15),dtype='float'),model.vectors),axis=0)
    return pos_dictionary, torch.tensor(pos_matrix)

def get_word_pos_label(fn,word_dictionary,pos_dictionary):
    f = open(fn)
    data = json.load(f)
    text = [ele['text'][:500] for ele in data]
    label = [ele['label'] for ele in data]
    word_ids = []
    pos_ids = []
    for ele in text:
        temp = []
        for token in ele:
            if token in word_dictionary:
                temp.append(word_dictionary[token])
            else:
                temp.append(1)
        if len(temp)<500:
            temp= [0]*(500-len(temp)) + temp
        word_ids.append(temp)
        temp = []
        tags = pos_tag(ele)
        for token in tags:
            if token[1] in pos_dictionary:
                temp.append(pos_dictionary[token[1]])
            else:
                temp.append(1)
        if len(temp)<500:
            temp= [0]*(500-len(temp)) + temp
        pos_ids.append(temp)

    return torch.tensor(word_ids), torch.tensor(pos_ids), torch.tensor(label)

word_dictionary, word_matrix = get_word_dictionary_matrix()
pos_dictionary, pos_matrix = get_pos_dictionary_matrix()
train_word_ids, train_pos_ids, train_label = get_word_pos_label('./train.json',word_dictionary, pos_dictionary)
dev_word_ids, dev_pos_ids, dev_label = get_word_pos_label('./dev.json',word_dictionary, pos_dictionary)
test_word_ids, test_pos_ids, test_label = get_word_pos_label('./test.json',word_dictionary, pos_dictionary)

with open('/scratch/near/anlp/word_dictionary.json','w') as f:
    f.write(json.dumps(word_dictionary))
with open('/scratch/near/anlp/pos_dictionary.json','w') as f:
    f.write(json.dumps(pos_dictionary))

torch.save(word_matrix,'/scratch/near/anlp/word2vec_matrix.pt')
torch.save(pos_matrix,'/scratch/near/anlp/pos2vec_matrix.pt')

torch.save(train_word_ids,'/scratch/near/anlp/train_word_ids.pt')
torch.save(train_pos_ids,'/scratch/near/anlp/train_pos_ids.pt')
torch.save(train_label,'/scratch/near/anlp/train_label.pt')

torch.save(dev_word_ids,'/scratch/near/anlp/dev_word_ids.pt')
torch.save(dev_pos_ids,'/scratch/near/anlp/dev_pos_ids.pt')
torch.save(dev_label,'/scratch/near/anlp/dev_label.pt')

torch.save(test_word_ids,'/scratch/near/anlp/test_word_ids.pt')
torch.save(test_pos_ids,'/scratch/near/anlp/test_pos_ids.pt')
torch.save(test_label,'/scratch/near/anlp/test_label.pt')
