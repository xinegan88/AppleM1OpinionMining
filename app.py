import sys
sys.path.insert(1, '/Users/christineegan/AppleM1SentimentAnalysis/src')

import general_functions
import twitter_api
import reddit_api
import process_data
import eda_visualizations
import word_vector_functions
import model_functions

    
if __name__ == '__main__':
    
    general_functions.create_banner('Social Media Opinion Mining App')
    
    platforms = ['1', '2']
    print('')
    print('Choose a platform: ')
    print('[Twitter [1], Reddit[2]]')
    platform = input()

    # choose a platform
    while platform not in platforms:
        print('\nInput not recognized. Try again.')
        platform = input()

    if platform == '1':
        general_functions.create_banner('Retrieve Tweets with Twython')
        answers = ['Y', 'N']
        print('\nWould you like to execute a schedule?', answers)
        answer = input()
        
        while answer not in answers:
            print('\nInput not recognized. Try again.')
            answer = input()
            
        if answer == answers[0]:
            print('\n[*] Launching Schedule...')
            general_functions.execute_schedule('Twitter')
            print('\nWould you like to preprocess the data?', answers)
            answer = input()
            
            while answer not in answers:
                print('\nInput not recognized. Try again.')
                answer = input()
                
            if answer == answers[0]:
                preprocess_data.batch_and_process_data('Twitter')
                
            elif answer == answers[1]:
                print('\n[*] Session Ended.')             

        else:
            q, result_type, count = twitter_api.tweet_params()
            query = twitter_api.tweet_query_summary(q, result_type, count)
            twitter_api.retrieve_tweets(query)
            
            print('\nWould you like to preprocess the data?', answers)
            answer = input()
            
            while answer not in answers:
                print('\nInput not recognized. Try again.')
                answer = input()
                
            if answer == answers[0]:
                preprocess_data.batch_and_process_data('Twitter')
                
            elif answer == answers[1]:
                print('\n[*] Session Ended.')

    elif platform == '2':
        GeneralFunctions.create_banner('Retrieve Reddit Submissions and Comments with PRAW')

        answers = ['Y', 'N']
        print('\nWould you like to execute a schedule?', answers)
        answer = input()
        
        while answer not in answers:
            print('\nInput not recognized. Try again.')
            answer = input()
            
        if answer == answers[0]:
            print('\n[*] Launching Schedule...')
            general_functions.execute_schedule('Reddit')
            print('\nWould you like to preprocess the data?', answers)
            answer = input()
            
            while answer not in answers:
                print('\nInput not recognized. Try again.')
                answer = input()
                
            if answer == answers[0]:
                preprocess_data.batch_and_process_data('Reddit')
                
            elif answer == answers[1]:
                print('\n[*] Session Ended.')

        # conduct one session
        else:
            term, limit, subreds = submissions_params()
            query = reddit_query_summary(term, limit, subreds)
            retrieve_submissions(query)

            print('\nWould you like to preprocess the data?', answers)
            answer = input()
            
            while answer not in answers:
                print('\nInput not recognized. Try again.')
                answer = input()
                
            if answer == answers[0]:
                preprocess_data.batch_and_process_data('Reddit')
                
            elif answer == answers[1]:
                print('\n[*] Session Ended.')