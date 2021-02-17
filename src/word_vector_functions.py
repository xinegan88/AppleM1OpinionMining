import pandas as pd
import numpy as np

import re

import os
from os import listdir
from os.path import isfile, join

from datetime import datetime

from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

import general_functions
import eda_visualizations


def create_vocabulary(df: pd.DataFrame) -> (list):
    '''Accepts a pd.DataFrame, iterates over the cleaned_text column and
    generates a vocabulary by making a list of every unique word.
    '''
    counts = eda_visualizations.count_words(df)
    
    return list(counts.keys())


def get_glove_vectors(total_vocab: pd.DataFrame, file_name: str) -> (dict):
    '''Navigates to the provided GLOVE file, opens it, and retrieves the
    word vectors for each word in the provided vocabulary pd.DataFrame.
    Each word and it's GLOVE vector are stored as a key-value pair in a 
    a dictionary.
    '''
    glove = {}
    with open(file_name, 'rb') as f:
        for line in f:
            parts = line.split()
            word = parts[0].decode('utf-8')
            if word.replace('^b', '') in total_vocab:
                vector = np.array(parts[1:], dtype=np.float32)
                glove[word] = vector
                
    return glove


def get_embedding(vectors: list, word: str) -> (list):
    '''Accepts a list of words and their GLOVE vectors as tuples, and 
    a word and returns a vector for the pair the matches the word.
    '''
    return [i[1] for i in vectors if i[0] == word]


def retrieve_glove_embeddings(model_data: pd.DataFrame, date_dir: str) -> (pd.DataFrame):
    '''Creates a vocabulary for the provided model data, then it imports 
    GLOVE vectors for the provided vocabulary. Next it tags each doc
    in the cleaned text data. Then, it initializes a Doc2Vec model and
    infers the vectors fo each word in the cleaned text. Finally, the
    words and vectors are zipped together, and the column for each word
    in the vocabulary is populated by the vector for that word in the
    rows in which the word is located within the cleaned text. Then, the 
    unneccessary features for the model data are dropped and and a time-
    stamped batch name is established. The vectorized model data is then
    saved to the designated directory for that batch. 
    '''
    general_functions.create_banner('Vectorize Text with Doc2Vec')

    print('\n[*] Creating vocabulary...')
    total_vocab = create_vocabulary(model_data)
    data = model_data.reset_index().drop('index', axis=1)

    print('\n[*] Importing GLOVE...')
    file_name = '/Users/christineegan/TwitterSentimentAnalysis/data/glove/glove.6B.100d.txt'
    glove = get_glove_vectors(total_vocab, file_name)

    print('\n[*] Tagging Documents')
    documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(data.cleaned_text)]

    print('\n[*] Creating model...')
    model = Doc2Vec(documents, window=2, min_count=1, workers=4)

    print('\n[*] Obtaining word vectors...')
    data['cleaned_text_new'] = data.cleaned_text.apply(lambda x: re.sub('[^0-9a-zA-Z ]+', '', x))
    data['cleaned_text_new'] = data.cleaned_text_new.apply(lambda x: x.split(' '))
    data['vex'] = data.cleaned_text_new.apply(lambda x: model.infer_vector(x))

    print('\n[*] Processing word vectors from text...')
    for word in total_vocab:
        data[word] = list(zip(data.cleaned_text_new, data.vex))
        data[word] = data[word].apply(lambda x: list(zip(x[0], x[1])))
        data[word] = data[word].apply(lambda x: get_embedding(x, word))
        data[word] = data[word].apply(lambda x: x[0] if len(x) > 0 else 0)

    data = data.drop(['text', 'cleaned_text', 'text_len', 'pos_tags', 'vex',
                      'cleaned_text_new', 'word', 'frequency'], axis=1)
        

    print('\n[*] Preparing data for model...')
    cols = [i for i in data.columns if i.isalpha()]
    data = data[cols]
    
    batch_name = ' '.join(str(datetime.now()).split(' '))[0:19]
    batch_name = batch_name.replace(':', '_').replace(' ','_')+'.csv' 
    file_name = date_dir + batch_name
    
    data.to_csv(file_name, index=False)
    print('[*] Saving data to', file_name)
    
    return data