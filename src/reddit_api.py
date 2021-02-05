import pandas as pd
import numpy as np

import sys
import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

from datetime import datetime

import general_functions

import praw
from praw.models import MoreComments

client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
user_agent = os.getenv('user_agent')

reddit = praw.Reddit(
     client_id=client_id,
     client_secret=client_secret,
     user_agent=user_agent)


def submissions_params():
    
    print('\nEnter a search term.')
    term = input()
    print('\nEnter a limit.')
    limit = input()
    print('\nEnter a list of subreddits seperated by commas.')
    subreds = input()
#     print(subreds)
#     print(type(subreds))
    subreds = subreds.split(', ')
#     print(type(subreds))
#     print(subreds)
    
    return term, limit, subreds


def reddit_query_summary(term, limit, subreds):

    print('')
    general_functions.create_banner('Summary of Query')
    print('Check your responses carefully to avoid wasting API calls.')
    
    print('')
    print('='*30)
    print('Search term: ', term.replace('+', ' ').lower())
    print('Limit: ', limit)
    print('Subreddits:', subreds)
    print('='*30)


    answers = ['Y','N']
    print('\nAre you happy with your query?', answers)
    answer = input()
    
    while answer not in answers:
        print('Response not recogized. Please try again.')
        answer = input()
        
    if answer.upper() == answers[0]:
        print('\n[*] Generating Query...')
        
        return term, limit, subreds
    
    elif answer.upper() == answers[1]:
        print('\nPlease re-enter your parameters.')
        term, limit, subreds = submissions_params()
        
        return term, limit, subreds
    
    
def make_reddit_query(term, limit, subred):

    sub_list =[]
    print('\n[*] Searching for [', term, '] in [r/', subred, ']...' , sep='')
    for submission in reddit.subreddit(subred).search(term, sort='comments', limit=int(limit)):
        subs = []
        sub = {}
        sub['title'] = submission.title
        sub['selftext'] = submission.selftext
        sub['subred'] = subred
        subs.append(sub)
        
    sub_list.append(subs)
    
    subs_df = pd.DataFrame()
    for sub in sub_list:
        df = pd.DataFrame(sub)
        df['text'] = df.title + ' ' + df.selftext
        df = df.drop(['title', 'selftext'], axis=1)
        subs_df = pd.concat([subs_df, df], axis=0)
    

    all_comments_list = []
    print('\n[*] Searching for [', term, '] in comments for [r/', subred, ']...' , sep='')
    for submission in reddit.subreddit(subred).hot(limit=20):
    
        comment_list = []
        for top_level_comment in submission.comments:
            if isinstance(top_level_comment, MoreComments):
                continue
            
            c = top_level_comment.body
            comment_list.append(c)
        
        all_comments_list.append(comment_list)

    comment_df = pd.DataFrame()
    for com in all_comments_list:
        df = pd.DataFrame(com)
        comment_df = pd.concat([comment_df, df], axis=0)
        if 0 in list(comment_df.columns):
            comment_df['text'] = comment_df[0]
            comment_df = comment_df.drop(0, axis=1)

    df = pd.concat([subs_df, comment_df])
    
    return df


def retrieve_submissions(term, limit, subreds):
    
    results = pd.DataFrame()
    directory = '~/AppleM1SentimentAnalysis/data/reddit_data/raw_data/subred_data/'
    filename = ' '.join(str(datetime.now()).split(' '))[0:19].replace(':', '_').replace(' ','_')+'.csv'
    csv_name = directory + filename
    
    for subred in subreds:
        print('\n[*] Retrieving data from r/' + subred)
        subred_df = make_reddit_query(term, limit, subred)
        
        print('\n[*] Saving results to ', csv_name)
        subred_df.to_csv(subred + ' ' + filename)
    
    return