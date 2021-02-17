import pandas as pd
import numpy as np

import shutil

import sys

import os
from os import listdir
from os.path import isfile, join

from datetime import datetime

import re
import string

from textblob import TextBlob

import nltk
from nltk import pos_tag, pos_tag_sents
from nltk.tokenize import TweetTokenizer 
tokenizer = TweetTokenizer()

from nltk.corpus import stopwords

additional_stops = ['apple', 'm1', 'mac', 'new', 'rt', 'get', 'go', 'one', 'even', 'would',
'macs', 'make', 'want', 'yes', 'really', 'could', 'say', 'lot', 'via', 'something', 'right',
'since', 'give', 'hackintosh', 'ago', 'hi', 'ask', 'bo', 'probably', 'put', 'end', 'might', 
'around' 'us', 'happen', 'kill', 'use', 'mini', 'macbook']

stopwords_list = stopwords.words('english')
stopwords_list += list(string.punctuation + string.digits)
stopwords_list += additional_stops

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

import general_functions


def clean_text(text: str, tokenizer: str) -> (str):
    '''Accepts a lemmatizer, text, stopwords. Tokenizes the text and 
    removes usernames, hashtags, and web addresses. Then it lemmatizes 
    the text and normalizes to lowercase. Returns a list of modified text
    tokens.
    '''
    text = tokenizer.tokenize(text)
    text = [t for t in text if t.isalpha()]
    text = [t.lower() for t in text]
    remove = [t for t in text if t.startswith('@') or t.startswith('#')]
    remove += [t for t in text if t.startswith('http') or t.startswith('www')]
    tokens = [t.replace('\n', '') for t in text if t not in remove]
    
    return ' '.join(tokens)


def process_text(text: str, stopwords_list: list, 
                 lemmatizer: str) -> (list):
    '''Accepts text, stopwords, and a lemmatizer, removes stopwords and
    lemmatizes text.
    '''
    text = text.split(' ')    
    tokens = [t for t in text if t not in stopwords_list]
    
    return [lemmatizer.lemmatize(token, pos='v') for token in tokens]


def label_subjectivity(df: pd.DataFrame) -> (pd.DataFrame):
    '''Accepts a dataf frame witha  cleaned text features, and applies a
    TextBlob subjectivity score.
    '''
    df['scores'] = df['cleaned_text'].apply(lambda x: TextBlob(x).sentiment)
    df['subjectivity'] = df.scores.apply(lambda x: x[1])
    df = df.drop('scores', axis=1)
    
    return df


def label_polarity(df: pd.DataFrame, analyzer: str) -> (pd.DataFrame):
    '''Accepts a dataf frame witha  cleaned text features, and calculates
    a set of polarity scores. The polarity scores is divided, and all 
    scores are dropped except for the compound score. The compound score is
    classified as -1, 0, or 1.
    '''
    df['scores'] = df.cleaned_text.apply(lambda x: 0 if type(x) == float else analyzer.polarity_scores(x))
    df['pos'] = df.scores.apply(lambda x: x['pos'])
    df['neg'] = df.scores.apply(lambda x: x['neg'])
    df['neu'] = df.scores.apply(lambda x: x['neu'])
    df['com'] = df.scores.apply(lambda x: x['compound'])
    df['polarity'] = df.com.apply(lambda x: 1 if x > 0 else x)
    df['polarity'] = df.polarity.apply(lambda x: -1 if x < 0 else x)
    df = df.drop(['scores', 'pos', 'neg', 'neu'], axis=1)
    
    return df


def count_tags(text: list, tag: str) -> (list):
    '''Accepts a list of a tuples that contain a word and its POS tag.
    Returns a list of only the tags.
    '''
    return [t[1] for t in text if t[1] == tag]


def pos_tag_data(df: pd.DataFrame) -> (pd.DataFrame):
    '''Accepts a pd.DataFrame with a POS tag feature and applies
    NLTK pos_tag_sents to the values. Compiles a list of tags used
    to create a unique column for each POS tag for dummy encoding.
    Returns the transformed data frame.
    '''    
    df['pos_tags'] = pos_tag_sents(df['cleaned_text'].tolist())

    all_tags = []
    for tag in df.pos_tags:
        all_tags += [t[1] for t in tag if t[1] not in all_tags]

    for tag in all_tags:
        df[tag] = df.pos_tags.apply(lambda x: count_tags(x, 
                              tag)).apply(lambda x: len(x))
        
    return df


def process_data(df: pd.DataFrame, tokenizer: str, stopwords_list: list,
                 lemmatizer: str, analyzer: str) -> (pd.DataFrame):
    '''
    Accepts a pd.DataFrame and parameters. Applies, tokenization to
    the text data. Removes stopwords. Then analyzes the text to apply
    sentiment labels (subjectivity and polarity). Next, text is lemmatized.
    The transformed data frame is returned.
    ''' 
    print('\n[*] Intiating text cleaning...')
    print('-- Tokenizing...')
    print('-- Removing non-alphabetic characters...')
    print('-- Converting to lowercase...')
    print('-- Removing hashtags, web addresses, and mentions...')
    print('-- Removing any code tags...')
    print('-- Joining tokens for further processing...')
#     df['text'] = df['text'].apply(lambda x: str(x))
    df['cleaned_text'] = df['text'].apply(lambda x: clean_text(str(x), tokenizer))
    df['lens'] = df['cleaned_text'].apply(lambda x: len(x))
    df = df[df.lens > 0]
    cols = ['text', 'cleaned_text']
    drop_cols = [col for col in list(df.columns) if col not in cols]
    df = df.drop(drop_cols, axis=1)
    
    print('\n[*] Labeling subjectivity...')
    print('-- Calculating subjectivity scores...')
    df = label_subjectivity(df)
    
    print('\n[*] Labeling polarity...')
    print('-- Calculating polarity scores...')
    print('-- Determining polarity label...')
    df = label_polarity(df, analyzer)
    
    answers = ['Y', 'N']
    print('Would you like to eliminate the neutral class?', answers)
    answer = input()
    
    while answer.upper() not in answers:
        print('Input not recognized. Please try again.')
    if answer.upper() == answers[0]:
        df = df[df.polarity != 0]

    print('\n[*] Processing text...')
    print('-- Removing stopwords...')
    print('-- Lemmatizing tokens...')
    df['cleaned_text'] = df.cleaned_text.apply(lambda x: process_text(x, stopwords_list, lemmatizer))
    
    print('\n[*] Calculating text length...')
    df['text_len'] = df['text'].apply(lambda x: len(x.split(' ')))
    
    print('\n[*] Applying POS tags...')
    print('-- Obtaining POS tags...')
    print('-- Creating POS tag list...')
    print('-- Counting POS tags...')
    df = pos_tag_data(df)
    
    print('\n[*] Preprocessing Complete')
    
    return df


def batch_and_process_data(platform: str) -> (pd.DataFrame):
    '''Data for the provided platform is retrieved and batched. The batch 
    is saved in its raw form, and then processed and returned.
    '''
    general_functions.create_banner('Preprocess Data')
    
    ignore = ['.DS_Store']
    source = '/Users/christineegan/AppleM1SentimentAnalysis/data/'
    time_stamp = str(datetime.date(datetime.now()))+'/'
    t_source_dir = source + 'tweet_data/raw_data/' + time_stamp
    t_target_dir = source + 'tweet_data/labeled_data/model_data/'
    r_source_dir = source + 'reddit_data/raw_data/session_data/' + time_stamp
    r_target_dir = source + 'data/reddit_data/labeled_data/model_data/'

    if platform == 'Twitter':
        print('[*] Retrieving session data from source directory...\n')
        file_names, raw_data, batch_name = general_functions.retrieve_batch(t_source_dir, ignore)
        target_dir = t_target_dir

    elif platform == 'Reddit':
        print('[*] Retrieving session data from source directory...\n')
        file_names, raw_data, batch_name = general_functions.retrieve_batch(r_source_dir, ignore)
        target_dir = r_target_dir
        
    else:
        print('[*] Retrieving session data from source directory...\n')
        t_file_names, t_raw_data, t_batch_name = general_functions.retrieve_batch(t_source_dir, ignore)
        r_filenames, r_raw_data, r_batch_name = retrieve_batch(t_source_dir, ignore)
        raw_data = pd.concat([twitter_raw_data, reddit_raw_data], axis=0)
        batch_name = ' '.join(str(datetime.now()).split(' '))[0:19]
        batch_name = batch_name.replace(':', '_').replace(' ','_')+'.csv'  
        target_dir = '/Users/christineegan/AppleM1SentimentAnalysis/data/combined_data/'
        
    print('\n[*] Preprocessing batch:', batch_name)
    data = process_data(raw_data, tokenizer, stopwords_list, lemmatizer, analyzer)
    
    date_dir = general_functions.date_directory(target_dir)
    print(date_dir)

    file_name = ' '.join(str(datetime.now()).split(' '))[0:19]
    file_name = file_name.replace(':', '_').replace(' ','_')+'.csv'
    csv_name = date_dir + file_name 
    
    print('\n[*] Saving processed results to ', csv_name)
    data.to_csv(csv_name, index=False)
        
    return data