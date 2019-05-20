import json
import spacy
from random import sample
import time
tokenizer = spacy.load("en_core_web_sm")
f = open('/scratch/near/review.json')

def parse_line(line):
    item = json.loads(line)
    words = tokenizer(' '.join(item['text'].split(' ')[:500]))
    words = [str(ele).lower() for ele in words]
    star = item['stars']
    return {'label':star, 'text':words}

star_list = []
for line in f:
    item = json.loads(line)
    star = item['stars']
    star_list.append(star)
star_list = [(i,ele) for i,ele in enumerate(star_list)]
star_list= sorted(star_list, key=lambda x: x[1])
dev_test_pool = sample(star_list,4000)
star_list = set(star_list)- set(dev_test_pool)
train_pool = []
for star in range(1,6):
    train_pool.extend(sample([ele[0]
                              for ele in star_list
                              if ele[1] == star],
                             100000))
dev_pool = dev_test_pool[:2000]
test_pool = dev_test_pool[2000:]


f = open('/scratch/near/review.json')
train_data = []
dev_data = []
test_data = []
train_set = set(train_pool)
dev_set = set([ele[0] for ele in dev_pool])
test_set = set([ele[0] for ele in test_pool])
for i,line in enumerate(f):
    if i %1000 == 0:
        print(i,time.ctime())
    if i in train_set:
        train_data.append(parse_line(line))
    elif i in dev_set:
        dev_data.append(parse_line(line))
    elif i in test_set:
        test_data.append(parse_line(line))

with open('./train.json','w') as f_train:
    f_train.write(json.dumps(train_data))
    print('write train json')
with open('./dev.json','w') as f_dev:
    f_dev.write(json.dumps(dev_data))
    print('write dev json')
with open('./test.json','w') as f_test:
    f_test.write(json.dumps(test_data))
    print('write test json')
