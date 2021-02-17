import pandas as pd
import numpy as np

from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns


def sentiment_frequency(df: pd.DataFrame) -> (None):
    ''''Accepts a pd.DataFrame containing polarity data and generates a
    bar plot of the frequency of each polarity class.
    '''
    
    sns.set()
    sns.set_style('dark')
    sns.set_palette('twilight')
    sns.set_style('ticks', {"xtick.major.size":12,"ytick.major.size":5})
    sns.set_context('paper', font_scale=1.5, rc={"lines.linewidth":2.5})

    val_counts = pd.DataFrame(df.polarity.value_counts())
    val_counts = val_counts.reset_index()
    val_counts = val_counts.rename(columns={'polarity': 'frequency'})
    val_counts = val_counts.rename(columns={'index': 'polarity'})
    ax1 = sns.barplot(x='polarity', y='frequency', data=val_counts)
    ax1.set_title('Class Frequency')
    
    plt.show();
    
    return


def divide_df(df: pd.DataFrame) -> (pd.DataFrame, 
                                    pd.DataFrame, 
                                    pd.DataFrame):
    '''Accepts a pd.DataFrame that contains polarity as a feature and 
    returns three pd.DataFrames divided by polarity.
    '''   
    print('\n[*]Dividing data frame by polarity...')
    pos_df = df[df['polarity'] == 1]
    neu_df = df[df['polarity'] == 0]
    neg_df = df[df['polarity'] == -1]
    
    return pos_df, neu_df, neg_df


def count_words(df: pd.DataFrame) -> (pd.DataFrame):
    '''Accepts df (pd.DataFrame), and adds each word in the column to a 
    list. A counter object is instantiate and each unique word and its 
    corresponding count are stored in a pd.DataFrame. Returns a
    a pd.DataFrame of words and frequency sorted by frequency.
    ''' 
    word_list = []
    for row in df.cleaned_text:
        row = row.replace('[', '').replace(']', '')
        row = row.replace("\'", '').replace(',', '')
        row = row.split(' ')
        words = [w for w in row]
        word_list += words

    from collections import Counter
    counter = Counter(word_list)
    counts = pd.DataFrame.from_dict(counter, orient='index',
                        columns=['frequency']).reset_index()
    counts = counts.rename(columns={'index': 'word'}).sort_values(by='frequency',
                                                      ascending=False)
    
    return counts


def word_frequency_by_sentiment(data: list, 
                                titles: list, 
                                sup_title: str) -> (None):
    '''Accepts three pd.DataFrames, a list of titles for each dataframe, 
    and suptitle for each plot. 
    '''  
    sns.set()
    sns.set_style('dark')
    sns.set_palette('prism')
    sns.set_style('ticks', {"xtick.major.size":12,"ytick.major.size":5})
    sns.set_context('paper', font_scale=1.5, rc={"lines.linewidth":2.5})

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharey=True)
    fig.set_figheight(15)
    fig.set_figwidth(15)

    ax1 = plt.subplot(311)
    sns.barplot(x='word', y='frequency', data=data[0], ax=ax1)
    ax1.set_title(titles[0])
    ax1.set_xticklabels(ax1.get_xticklabels(),rotation=75)

    ax2 = plt.subplot(312)
    sns.barplot(x='word', y='frequency', data=data[1], ax=ax2)
    ax2.set_title(titles[1])
    ax2.set_xticklabels(ax2.get_xticklabels(),rotation=75)

    ax3 = plt.subplot(313)
    sns.barplot(x='word', y='frequency', data=data[2], ax=ax3)
    ax3.set_title(titles[2])
    ax3.set_xticklabels(ax3.get_xticklabels(),rotation=75)

    fig.tight_layout(h_pad=5)
    fig.suptitle(sup_title, position=(0.5, 1.025), fontsize=28, va='baseline');
    
    return


def create_wordcloud(counts: pd.DataFrame) -> (WordCloud):
    '''Accepts a pd.DataFrame of word counts and displays a word cloud 
    visualization.
    '''
    
    all_words = list(counts['word'])
    allwords = all_words[20:300]
    all_words = ' '.join(all_words)
        
    cloud = WordCloud(width=800, height=400, colormap='twilight_shifted_r',
                        random_state=21, max_font_size=110,
                        collocations=True).generate(all_words)
    
    return cloud


def wordclouds_by_sentiment(pos_df: pd.DataFrame, neu_df: pd.DataFrame,
                          neg_df: pd.DataFrame, titles: list) -> (None):
    '''Accepts 3 pd.DataFrames divided by sentiment, and a list of three 
    titles, and returns a figure with one plot for each dataframe.
    '''
    
    pos_counts = count_words(pos_df)
    neu_counts = count_words(neu_df)
    neg_counts = count_words(neg_df)
    
    counts = [pos_counts, neu_counts, neg_counts]

    fig = plt.figure()
    fig.set_figheight(20)
    fig.set_figwidth(20)
    fig.tight_layout()

    for i in range(len(counts)):
        ax = fig.add_subplot(1, len(counts), i+1)
        wordcloud = create_wordcloud(counts[i])

        ax.imshow(wordcloud, interpolation='bilinear')
        ax.set_title(titles[i], fontsize=20)
        ax.axis('off')

    fig.tight_layout(h_pad=5)
    fig.suptitle('Most Frequent Words by Sentiment', position=(0.5, 0.65),
                 fontsize=28, va='baseline');
    plt.show()
    
    return
    

def relationship_between_features(df: pd.DataFrame) -> (None):
    ''''Accepts a pd.DataFrame and returns a figure with three subplots 
    that represent the relationship between features in the dataframe.
    '''
    
    sns.set()
    sns.set_palette('twilight')
    sns.set_style('ticks', {"xtick.major.size":12,"ytick.major.size":5})
    sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth":2.5})

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
    fig.set_figheight(18)
    fig.set_figwidth(9)

    ax1 = plt.subplot(311)
    sns.scatterplot(data=df, x='lens', y='com', ax=ax1)
    ax1.set_title('Compound Polarity vs Text Length')

    ax2 = plt.subplot(312)
    sns.scatterplot(data=df, x='lens', y='subjectivity', ax=ax2)
    ax2.set_title('Subjectivity vs Text Length')

    ax3 = plt.subplot(313)
    sns.scatterplot(data=df, x='com', y='subjectivity', ax=ax3)
    ax3.set_title('Compound Polarity vs Subjectivity')

    fig.tight_layout(h_pad=5)
    fig.suptitle('Relationship Between Features', position=(0.5, 1.025), 
                 fontsize=28, va='baseline');
    
    plt.show();
    
    return


def pos_tag_counts(df: pd.DataFrame) -> (pd.DataFrame):
    '''Accepts a pd.DataFrame of POS tags in a sparse matrix and and returns a
    pd.DataFrame containg every unique POS tag and its frequency. '''
    
    pos_counts = df.drop(['text', 'cleaned_text', 'lens', 'subjectivity', 'com',  
                      'polarity', 'text_len', 'pos_tags'], axis=1)
    pos_counts_df = pd.DataFrame(pos_counts.sum(axis=0, skipna=True))
    pos_counts_df = pos_counts_df.rename(columns={'index': 'tag', 0 : 'frequency'})
    pos_counts_df = pos_counts_df.sort_values(by='frequency', ascending=False)[:10]
    pos_counts_df = pos_counts_df.reset_index().rename(columns={'index': 'tag'})
    total_tags = pos_counts_df.frequency.sum()
    
    pos_counts_df['percent'] = pos_counts_df.frequency.apply(lambda x: round(x*100 / total_tags, 1))
    
    return pos_counts_df


def pos_tag_frequency_by_sentiment(pos_df: pd.DataFrame, neu_df: pd.DataFrame,
                                   neg_df: pd.DataFrame) -> (None):
    ''''Accepts a pd.DataFrame and returns a figure
    with three subplots that represent the relationship
    between features in the dataframe.'''
    
    pos_counts = pos_tag_counts(pos_df)
    neu_counts = pos_tag_counts(neu_df)
    neg_counts = pos_tag_counts(neg_df)
    
    pos_counts = pos_counts.rename(columns={'tag': 'pos_tag', 
                                            'frequency': 'pos_freq', 
                                            'percent': 'pos_percent'}
                                              )
    neu_counts = neu_counts.rename(columns={'tag': 'neu_tag',
                                            'frequency': 'neu_freq',
                                            'percent': 'neu_percent'}
                                              )
    neg_counts = neg_counts.rename(columns={'tag': 'neg_tag', 
                                            'frequency': 'neg_freq', 
                                            'percent': 'neg_percent'}
                                              )
    
    all_counts = pd.concat([pos_counts, neu_counts, neg_counts], axis=1)
    
    pos_list = list(pos_counts['pos_tag'])
    neu_list = list(neu_counts['neu_tag'])
    neg_list = list(neg_counts['neg_tag'])

    all_list = [i for i in pos_list if i in neu_list and i in neg_list]
    
    print('\nShared POS Tags Among All Classes')
    print(all_list)
    
    unique_pos = [i for i in pos_list if i not in all_list]
    unique_neu = [i for i in neu_list if i not in all_list]
    unique_neg = [i for i in neg_list if i not in all_list]
    
    print('\nUnique Positive POS Tags')
    print(unique_pos)
    print('')
    print('\nUnique Neutral POS Tags')
    print(unique_neu)
    print('')
    print('\nUnique Negative POS Tags')
    print(unique_neg)
    print('')

    counts = [pos_counts, neu_counts, neg_counts]
    sns.set()
    sns.set_palette('twilight')
    sns.set_style('ticks', {"xtick.major.size":12,"ytick.major.size":5})
    sns.set_context('notebook', font_scale=1.5, rc={"lines.linewidth":2.5})

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 6), sharey=True)
    fig.set_figheight(18)
    fig.set_figwidth(9)

    ax1 = plt.subplot(311)
    sns.barplot(x='pos_tag', y='pos_freq', data=pos_counts.sort_values(by='pos_freq'), ax=ax1)
    ax1.set_title('Positive')

    ax2 = plt.subplot(312)
    sns.barplot(x='neu_tag', y='neu_freq', data=neu_counts.sort_values(by='neu_freq'), ax=ax2)
    ax2.set_title('Neutral')

    ax3 = plt.subplot(313)
    sns.barplot(x='neg_tag', y='neg_freq', data=neg_counts.sort_values(by='neg_freq'), ax=ax3)
    ax3.set_title('Negative')

    fig.tight_layout(h_pad=5)
    fig.suptitle('POS Tag Frequencies by Polarity', position=(0.5, 1.025), fontsize=28, va='baseline')
    
    plt.show();
    
    return