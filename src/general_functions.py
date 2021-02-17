import numpy as np
import pandas as pd

import datetime
from datetime import *
import time as t

import shutil

import sys

import os
from os import listdir
from os.path import isfile, join

import twitter_api
import reddit_api


def create_banner(title: str) -> (None):
    '''Accepts title(str) and displays a simple banner for the 
    application using the provided title.
    '''
    print('')
    print('='*100)
    print(title)
    print('-'*100)
    
    return


def date_directory(target_dir: str) -> (str):
    '''Checks the target directory to see if the date stamped directory exists
    for the current date. If no directory exists, it is created and the name of
    the new directory is returned.
    '''
    dir = os.path.join(target_dir+str(datetime.date(datetime.now())))
    if not os.path.exists(dir):
        os.mkdir(dir)
    return str(dir+'/')
            

def execute_schedule(platform: str) -> (None):
    '''For a given platform, the user is prompted to set a timer to make
    API calls. The user enters a start time in 24-hour time. The default
    is that the schedule executes the script in 15-minute intervals, with
    sixty seconds of rest between each query. Each interval is followed by
    a five minute break, the the script is executed again until the user
    specified end time.
    '''
    create_banner('Execute Schedule')
    print('Don\'t forget to mind your API call limits.')
    print('\nEnter your endtime using 24-hour format.')
    hour = input('Hour: ')
    minute = input('Minute: ')

    print('\n[*] Initializing timer at', datetime.now())
    start_time_of_day = datetime.combine(date.today(), time(0, 0, 0))
    end_time_of_day = datetime.combine(date.today(), 
                      time(int(hour), int(minute), 0))
    
    print('[*] Scheduled to end at: ', end_time_of_day)
    
    while True:
        if datetime.now() >= start_time_of_day:
                if platform == 'Twitter':
                    q, result_type, count = twitter_api.tweet_params()
                    query = twitter_api.tweet_query_summary(q, result_type, count)
                    
                    while datetime.now() <= end_time_of_day:
                        twitter_api.retrieve_tweets(query)
                        print('\n[*] Sleeping for 60 Seconds.')
                        t.sleep(60)
                        continue
                        
                    if datetime.now() >= end_time_of_day:
                        print('\n[*] Endtime Reached. Session Ended.')
                        break
                    
                elif platform == 'Reddit':
                    term, limit, subreds = reddit_api.submissions_params()
                    query = reddit_api.reddit_query_summary(term, limit, subreds)
                    
                    print('\n[*] Sleeping for 60 Seconds.')
                    t.sleep(60)
                    
                    while datetime.now() <= end_time_of_day:
                        reddit_api.retrieve_submissions(query)
                        print('\n[*] Sleeping for 60 Seconds.')
                        t.sleep(60)
                        continue
                        
                    if datetime.now() >= end_time_of_day:
                        print('\n[*] Endtime Reached. Session Ended.')
                        break
                break
                
    return
            
def retrieve_batch(source_dir, ignore) -> (list, pd.DataFrame, str):
    '''Retrieves a list of csv file names from source_dir(string) while 
    ignoring files named in ignore list. The resulting list of files is
    read in and concatenated into a batch. batch.
    '''
    file_names = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
    file_names = [f for f in file_names if f not in ignore]
    batch_name = ' '.join(str(datetime.now()).split(' '))[0:19]
    batch_name = batch_name.replace(':', '_').replace(' ','_')+'.csv'

    batch = pd.DataFrame()
    for file_name in file_names:
        data = pd.read_csv(source_dir + '/' + file_name)
        print('[*] Adding ', file_name, 'to batch: ', batch_name)
        batch = pd.concat([batch, data], axis=0)
        
    return file_names, batch, batch_name                  