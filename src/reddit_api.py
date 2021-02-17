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


def submissions_params() -> (str, int, list):
    '''Prompts the user to enter a search term, a result limit and 
    a list of subreddits to provide submission parameters for a 
    Reddit API call.
    '''
    
    print('\nEnter a search term.')
    term = input()
    
    print('\nEnter a limit.')
    limit = input()
    
    print('\nEnter a list of subreddits seperated by commas.')
    subreds = input()
    subreds = subreds.split(', ')

    
    return term, limit, subreds


def reddit_query_summary(term: str, limit: int, 
                         subred: str) -> (str, int, list):
    '''Accepts a search term, result limit and list of subreddits and 
    constructs a query summary so the user can correct any mistakes before
    passing the selections along to the final query parameters.
    '''

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
    

def make_reddit_query(term: str, limit: int, 
                      subred: str) -> (pd.DataFrame):
    '''Makes Reddit API calls by constructing a query for the subreddit 
    indicated. It limits the reults to the limit indicated and the term
    indicated. The result of each API call is stored in a pd.DataFrame. 
    '''

    sub_list =[]
    print('\n[*] Searching for [', term, '] in [r/', subred, ']...' , sep='')
    for submission in reddit.subreddit(subred).search(term, 
                         sort='comments', limit=int(limit)):
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


def retrieve_submissions(term: str, limit: int, 
                      subreds: str) -> (None):
    '''Accepts the user selections for term, limit, and subreds. A 
    destination for the results retrieved is established. The list of
    subreddits is iterated over and queried for the selected term, and
    limited to the provided limit. The result of each subreddit's query
    is saved in a time stamped file in the provided directory. All of
    the files are cocatenated into a batch and saved as a time stamped
    session file.'''
    
    target_dir = '/Users/christineegan/AppleM1SentimentAnalysis/data/reddit_data/session_data/'
    date_dir = general_functions.date_directory(target_dir)
    file_name = ' '.join(str(datetime.now()).split(' '))[0:19]
    file_name = file_name.replace(':', '_').replace(' ','_')+'.csv'
    csv_name = date_dir + file_name
    
    for subred in subreds:
        print('\n[*] Retrieving data from r/' + subred)
        subred_df = make_reddit_query(term, limit, subred)
        
        print('\n[*] Saving results to ', csv_name)
        subred_df.to_csv(csv_name)
    
    return