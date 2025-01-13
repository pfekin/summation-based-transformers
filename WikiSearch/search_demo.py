# -*- coding: utf-8 -*-
import numpy as np
from pathlib import Path
import pickle
from scipy.spatial import distance

model_file = 'data/wiki_model'
is_lower_case = True

print('Load model')
with open(model_file, 'rb') as f:
    freq, index2word, word2index, word_vector, titles, doc_vector = pickle.load(f)

while True:
    tokens = input('\nSearch: ')
    tokens = tokens.lower() if is_lower_case else tokens
    tokens = tokens.split()

    query = []
    for token in tokens:
        if word2index.get(token) is not None:
            query.append(word_vector[word2index[token]])
    query_vector = np.sum(query, axis=0).reshape(1, word_vector[0].shape[0])
    results = distance.cdist(query_vector, doc_vector, metric='cosine')[0] # cosine
    best_results = np.argsort(results)[:20]
    
    # show results
    for i in range(len(best_results)):
        idx = best_results[i]
        print('[{:>5.5}]   '.format(results[idx]) + titles[idx][:-1])
  


    
