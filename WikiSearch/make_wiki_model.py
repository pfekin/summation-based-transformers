# -*- coding: utf-8 -*-
import collections 
import pickle
import re
import numpy as np

def make_rnd_vectors(vocab_size, dim, one_dim, drange=1.0, seed=1):
    np.random.seed(seed)
    root = np.zeros(dim, dtype=np.float32) - drange
    root[0 : one_dim] = drange
    word_vector = np.empty((vocab_size, dim), dtype=np.float32)
    for i in range(vocab_size):
        word_vector[i] = np.random.permutation(root) 
    return word_vector

def load_titles(file_name):
    titles = []
    with open(file_name, "r", encoding="utf8") as f:
        for title in f:
            titles.append(title)
    return titles

source_file = 'data/wiki.txt'
mode_file = 'data/wiki_model'
title_file = 'data/wiki_title.txt'
min_freq = 20
dim_size = 512
dim_one_size = 256
drange = 1.0
seed = 1

print('Extract vocabulary from text file')
freq = collections.Counter()
with open(source_file, 'rb') as text_f:
    for i, article in enumerate(text_f):
        words = article.decode('utf-8').split()
        freq.update(words)
        if i % 10000 == 0:
            print(i, 'articles processed')

print('Vocab size', len(freq))
for key in dict(freq):
    if freq[key] < min_freq:
        del(freq[key])    
print('Vocab size after pruning', len(freq))

index2word = [word[0] for word in freq.most_common()]
word2index = dict(zip(index2word, range(len(index2word))))
word_vector = make_rnd_vectors(len(freq), dim_size, dim_one_size, drange, seed)

doc_vector = []
print('Make document vectors')
with open(source_file, 'rb') as text_f:
    for i, article in enumerate(text_f):
        words = article.decode('utf-8').split()
        article = np.zeros((word_vector.shape[1]))
        for word in words:
            if word2index.get(word) is not None:
                article += word_vector[word2index[word]] / np.log(freq[word])               
        doc_vector.append(article)
        if i % 10000 == 0:
            print(i, 'articles processed')
doc_vector = np.array(doc_vector, dtype=np.float32)

print('Save model i.e. freq, index2word, word2index, word_vector, titles, doc_vector')
with open(mode_file, 'wb') as f:
    pickle.dump([freq, index2word, word2index, word_vector, load_titles(title_file), doc_vector], f, pickle.HIGHEST_PROTOCOL)
    