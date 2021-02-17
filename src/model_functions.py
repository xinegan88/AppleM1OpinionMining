import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime

import sklearn
import sklearn.model_selection as model_selection
import sklearn.metrics as metrics
from imblearn.over_sampling  import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from word_vector_functions import retrieve_glove_embeddings
import general_functions


def create_banner(title: str) -> (None):
    '''Accepts title(str) and displays a simple banner for the
    application using the provided title.
    '''
    print('')
    print('='*100)
    print(title)
    print('-'*100)
    
    return


def import_model_data(file_name: str) -> (pd.DataFrame):
    '''Accepts a csv file as input, reads it in as a pd.DataFrame, and
    divides the data into X, y. Returns X, y.
    '''
    source = '/Users/christineegan/AppleM1SentimentAnalysis/data'
    if 'tweet' in file_name:
        target_dir = source + '/tweet_data/labeled_data/model_data/'
        date_dir = general_functions.date_directory(target_dir)
    else:
        target_dir = source + '/reddit_data/labeled_data/model_data/'
        date_dir = general_functions.date_directory(target_dir)
        
    print('\n[*] Importing model data...')
    model_data = pd.read_csv(file_name)
    model_data = model_data[model_data.polarity != 0]
    vectorized_model_data = retrieve_glove_embeddings(model_data, date_dir)
    X = vectorized_model_data.drop(['polarity'], axis=1)
    y = vectorized_model_data.polarity

    return vectorized_model_data, X, y


def check_class_imbalance(data: pd.Series) -> (bool):
    ''' Accepts a pd.Series and retrieves the value counts and stores them 
    in a pd.DataFrame. The total frequency of each value is then calculated
    and stored as a percent. If the percent of all the values do not match
    a class imbalance is indicated.
    '''
    create_banner('Class Distribution')
    print('[*] Reviewing target data...')
    val_counts = pd.DataFrame(data.value_counts())
    val_counts = val_counts.rename(columns={'polarity': 'frequency'}).reset_index()
    val_counts = val_counts.rename(columns={'index': 'polarity'})
    total_data = val_counts.frequency.sum()
    val_counts['percent_of_data'] = val_counts.frequency.apply(lambda x: x*100 / total_data)
    display(val_counts)

    answers = ['Y','N']
    if len(np.unique(val_counts.frequency)) != 1:
        print('A class imbalance was detected. Would you like to use SMOTE?', answers)
        answer = input()
        if answer.upper() == answers[0]:
            use_smote = True
        else:
            use_smote = False

    return use_smote


def plot_my_conf_matrix(conf_matrix: np.ndarray, 
                        ax: plt.subplot) -> (None):
    '''Accepts an sklearn.metrics confusion matrix, and axis number
    and displays an sns.heatmap of the confusion matrix.
    '''
                        
    sns.heatmap(conf_matrix, annot=True, 
                fmt=".3f", linewidths=.5, 
                cmap='Blues_r', ax=ax)
                        
    ax.set_xlabel('Predicted Label', fontsize=10)
    ax.set_ylabel('Actual Label', fontsize=10)
    ax.set_title('Confusion Matrix',fontsize=12)
                        
    return
                        

def plot_my_roc_curve(clf: str, 
                      X_test: np.ndarray, 
                      y_test: np.ndarray, 
                      ax: plt.subplot) -> (None):
    '''Takes in a classifier, training data, and axis numbers. Displays
    an ROC Curve/AUC Score plot for the classifier.
    '''
                        
    metrics.plot_roc_curve(clf, X_test, y_test, alpha=1, lw=2, ax=ax)
    ax.set_title('ROC Curve/AUC Score',fontsize=12)
    
    return
    

def base_model(clf: str, X: pd.DataFrame, y: pd.Series, test_size: float, 
               random_state: float, use_smote: bool) -> (None):
    '''Accepts an sklearn classifier, data, and parameters and executes a 
    base model for each test size (0.2-0.4). Displays the classification 
    report, ROC-Curve and confusion matrix. 
    '''
    print('Test Size: ', test_size)
    print('[*] Generating basic model with TTS for:', clf, '...')
    X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, 
                                  test_size=test_size, random_state=random_state)
    
    if use_smote == True:
        print('[*] Oversampling data with SMOTE')
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_sample(X_train, y_train)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    print('[*] Generating Classification Report...')
    print('\nClassification Report')
    print('='*100)
    print(metrics.classification_report(y_test,y_pred))
    print('-'*100)

    create_banner('Confusion Matrix and ROC Plot')
    conf_matrix = metrics.confusion_matrix(y_test, y_pred)    
    fig, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(12, 4))
    plot_my_conf_matrix(conf_matrix, ax1)
    plot_my_roc_curve(clf, X_test, y_test, ax2)
    plt.plot([0,1],[0,1], linestyle='--', lw=2, color='black')
    plt.show()
    print('')
    
    return
    

def run_cross_val_model(clf: str, 
                        X: np.ndarray, y: np.ndarray,
                        cv: sklearn.model_selection.StratifiedKFold) -> (None):
    '''Runs a cross validation model given a classifier, X, y data, and 
    cv parameters. Returns the model scores for each fold in the 
    cross validation.
    '''

    scores = pd.DataFrame()
    scores['Accuracy'] = model_selection.cross_val_score(clf, X, y, cv=cv,
                                                       scoring='accuracy')
    scores['AUC'] = model_selection.cross_val_score(clf, X, y, cv=cv, 
                                                    scoring='roc_auc')
    scores['Precision'] = model_selection.cross_val_score(clf, X, y, cv=cv, 
                                                       scoring='precision')
    scores['Recall'] = model_selection.cross_val_score(clf, X, y, cv=cv, 
                                                       scoring='recall')
    scores['F1'] = model_selection.cross_val_score(clf, X, y, cv=cv, 
                                                       scoring='f1')
    clf_scores = scores.sort_values(by='Precision').reset_index().drop('index', axis=1)
    clf_scores.loc['mean'] = scores.mean()
    
    display(clf_scores)
    
    return clf_scores, clf_scores.mean()


def plot_my_cross_val_roc_curve(clf: str, X: np.ndarray, y: np.ndarray,
                        cv: sklearn.model_selection.StratifiedKFold,
                        use_smote: bool) -> (None):
    '''Runs a cross validation model and returns a ROC curve with a plot for
    AUC results for each fold in the cross validation.
    ''' 
    fig = plt.figure(figsize=[8, 12])
    ax1 = fig.add_subplot(111, aspect='equal')
    
    true_pos_rates = []
    aucs = []
    mean_false_pos_rate = np.linspace(0, 1, 100)
    
    num = 1
    if use_smote == True:
        print('\n[*] Oversamping data with SMOTE...')
        
    print('\n[*] Plotting ROC...')
    for train, test in cv.split(X, y):
        if use_smote == True:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_sample(X.iloc[train], y.iloc[train])            

        y_pred = clf.fit(X.iloc[train], y.iloc[train]).predict_proba(X.iloc[test])
        false_pos_rate, true_pos_rate, t = metrics.roc_curve(y.iloc[test], y_pred[:, 1])
        true_pos_rates.append(np.interp(mean_false_pos_rate, 
                                        false_pos_rate, true_pos_rate))
        roc_auc = metrics.auc(false_pos_rate, true_pos_rate)
        aucs.append(roc_auc)
        plt.plot(false_pos_rate, true_pos_rate, lw=2, alpha=0.3, 
                 label='ROC fold %d (AUC = %0.2f)' % (num, roc_auc))
        num += 1

    plt.plot([0,1],[0,1],linestyle = '--',lw = 2,color = 'black')
    mean_true_pos_rate = np.mean(true_pos_rates, axis=0)
    mean_auc = metrics.auc(mean_false_pos_rate, mean_true_pos_rate)
    plt.plot(mean_false_pos_rate, mean_true_pos_rate, color='blue',
         label=r'Mean ROC (AUC = %0.2f )' % (mean_auc),lw=2, alpha=1)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()
    
    return
    

def model_results(clfs: list, X: pd.DataFrame, y: pd.Series,
                  use_smote: bool) -> (pd.DataFrame):
    '''Creates a pd.DataFrame of model scores by executing an a classifer
    model on X, y using both train, test, split in various test sizes
    and cross validation. Plots a confusion matrix and ROC/AUC curve, and
    returns the model scores.
    '''  
    model_scores = pd.DataFrame()
    num = 1
    for clf in clfs:
        test_sizes = [0.2, 0.3, 0.4]
        for test_size in test_sizes:
            base_model(clf, X, y, test_size, 42, use_smote)
            cv = sklearn.model_selection.StratifiedKFold(n_splits=5, 
                                      random_state=42, shuffle=True)
            
        clf_scores, clf_scores_mean = run_cross_val_model(clf, X, y, cv)
        plot_my_cross_val_roc_curve(clf, X, y, cv, use_smote)
        
    create_banner('Model Scores')
    print('\n[*] Calculating model scores...')
    model_scores = pd.concat([clf_scores, clf_scores_mean], axis=0)
    display(model_scores)
            
    return


def execute_models() -> (None):
    '''Model data is imported from the model data directory. The user is
    prompted to choose a model(s) and enter a target feature. The data is
    checked for a class imbalance and the user is prompted to choose the
    SMOTE preference. The model data is then split into X, y and is passed
    along with the user entered parameters to construct the final models.
    
    '''
    create_banner('Deploy a Model')
    print('Enter a file name.')
    file_name = input()
    vectorized_data, X, y = import_model_data(file_name)

    print('\n Choose a model: ')
    model_types = {'1': 'Logistic Regression', '2': 'GaussianNB', '3': 'SVC'}
    print(model_types)
    print('\n To choose more than one model, enter each choice seperated by a comma.')
    model_choices = input()
    
    for model in model_choices:
        while model not in list(model_types.keys()):
            print('Input not recognized, please try again.')
            model_choices = input()
    
    model_choices = [model_choices]
    if len(model_choices) > 1:
        model_choices = model_choices.split(' ')

    clf1 = LogisticRegression()
    clf2 = GaussianNB()
    clf3 = SVC()
    model_types = {'1': clf1, '2': clf2, '3': clf3}
    
    models = []
    for model in model_choices:
        models.append(model_types[model])
    
    use_smote = check_class_imbalance(y)
    
    create_banner('Model Results')
    X = vectorized_data.drop('polarity', axis=1)
    y = vectorized_data.polarity
    model_results(models, X, y, use_smote)

    print('\nSession Ended')
    return
    