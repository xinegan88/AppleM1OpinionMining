import pandas as pd
import numpy as np

import sys

import os
from dotenv import load_dotenv
load_dotenv(verbose=True)

import re

from twython import Twython

import time as t
from datetime import datetime

import general_functions


def tweet_params() -> (str, str, int):
    '''A script that collects user input to generate parameters for the 
    Twitter API Recent Search endpoint. User will enter search query, 
    result type, and number of results. Specified params are passed on
    to created a query summary.
    '''
    print('\nEnter a search term.')
    term = input()
    term = term.replace(' ', '+').lower()

    print('\nEnter your result type:')
    result_types = {'1': 'recent', '2': 'popular', '3': 'mixed'}
    print(result_types)
    r_type = input()
    
    while r_type not in list(result_types.keys()):
        print('Result type not recognized, please try again.')
        r_type = input()
    
    result_type = result_types[r_type]

    print('\nEnter a result count: ')
    count = input()
    count = int(count)
    
    while count > 100:
        print('Too many. Enter a smaller number.')
        count = input()
        continue
        
    return term, result_type, count


def make_twython_query(term: str, result_type: str,
                       count: int) -> (dict):
    '''Accepts user paramaters and returns a Twython Query.'''
    query = {'q': term,
             'result_type': result_type,
             'lang': 'en',
             'count': count
            }
    
    return query


def tweet_query_summary(term: str, result_type: str,
                       count: int) -> (None):
    '''Accepts Twitter API Recent Search parameters,. Prints a summary of
    these parameters. Confirms with the user that the query is acceptable.
    If confirmed, query is passed along to create the final query. If not
    confirmed, user is prompted to re-enter parameters.
    '''
    print('')
    general_functions.create_banner('Summary of Query')
    print('Check your responses carefully to avoid wasting API calls.')
    
    print('')
    print('='*30)
    print('Search term: ', term.replace('+', ' ').lower())
    print('Result type: ', result_type)
    print('Number of results:', count)
    print('='*30)


    answers = ['Y','N']
    print('\nAre you happy with your query?', answers)
    answer = input()
    
    while answer not in answers:
        print('Response not recogized. Please try again.')
        answer = input()
           
    if answer.upper() == answers[0]:
        query = make_twython_query(term, result_type, count)
        print('\n[*] Generating Query...')
        return query
    
    elif answer.upper() == answers[1]:
        print('\nPlease re-enter your parameters.')
        term, result_type, count = tweet_params()
        
    return


def twython_results(api_key: str, api_key_secret: str, 
                    query: dict) -> (pd.DataFrame):
    '''Accepts users credentials and approved query, executes a Twitter
    search for the query, and returns the results in a pd.DataFrame that
    is filtered for the designated features.
    '''

    print('\n[*] Obtaining authorization...')
    twitter = Twython(api_key, api_key_secret)
    auth = twitter.get_authentication_tokens()

    OAUTH_TOKEN = auth['oauth_token']
    OAUTH_TOKEN_SECRET = auth['oauth_token_secret']

    print('\n[*] Filtering results...')
    dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}

    for status in twitter.search(**query)['statuses']:
        dict_['user'].append(status['user']['screen_name'])
        dict_['date'].append(status['created_at'])
        dict_['text'].append(status['text'])
        dict_['favorite_count'].append(status['favorite_count'])
    
    results = pd.DataFrame(dict_)

    return results


def retrieve_tweets(query: dict) -> (None):
    '''Establishes a time stamped filename and directory to store
    retults. Provided credentials are retrieved and provided to
    execute the API calls for the query. A csv of the results is
    saved in the established file name.
    '''
    
    target_dir = '/Users/christineegan/AppleM1SentimentAnalysis/data/tweet_data/raw_data/'
    date_dir = general_functions.date_directory(target_dir)
    file_name = ' '.join(str(datetime.now()).split(' '))[0:19]
    file_name = file_name.replace(':', '_').replace(' ','_')+'.csv'
    csv_name = date_dir + file_name

    print('\n[*] Accessing credentials...')
    api_key = os.getenv('twitter_api_key')
    api_key_secret = os.getenv('twitter_api_secret_key')
    
    results = twython_results(api_key, api_key_secret, query)
    
    print('\n[*] Saving results to ', csv_name)
    results.to_csv(csv_name, index=False)

    return


def retrieve_more_tweets() -> (None):
    '''Prompts the user to retrieve more tweets or
    end the session.
    '''
    answers = ['Y', 'N']
    print('\nWould you like to retrieve more tweets?', answers)
    answer = input()
    
    while answer.upper() not in answers:
        print('Response not recogized. Please try again.')
        answer = input()
        
    if answer.upper() == answers[0]:
        retrieve_tweets()
        
    elif answer.upper() == answers[1]:
        print('Session Ended')
        
    return