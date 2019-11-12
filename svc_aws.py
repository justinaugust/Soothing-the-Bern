import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.compose import ColumnTransformer
from tokenizer.tokenizer import RedditTokenizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


import time

from my_tools.pipes_grids import *

nltk_stops = set(stopwords.words('english'))
df = pd.read_csv('datasets/data_submissions.csv')
features_txt = ['body']
features_non_txt = list(df.drop(columns=features_txt+['score','is_2016','senti_score']).columns.values)
features = features_txt + features_non_txt
ct = ColumnTransformer(
    [
        ('tvec',TfidfVectorizer(), 'body'),
    ],
                  remainder='passthrough',
                  sparse_threshold=0.3,
                  n_jobs=-1,
                  )



pipe = Pipeline([
    ('ct',ct),
    ('svc',SVC(gamma='scale'))

])

pipe_params = {
    'ct__tvec__tokenizer' : [None,RedditIt()],
    'ct__tvec__ngram_range' : [(1,1),(1,2)],
    'ct__tvec__max_features' : [None],
    'ct__tvec__min_df' : [.01,.02],
    'ct__tvec__max_df' : [.9,1.0],
    'ct__tvec__stop_words' : [nltk_stops],
    'svc__C' : np.logspace(-2,2, 30),
    'svc__kernel' : ['linear','rbf','poly'],







}

gs = GridSearchCV(pipe, # what object are we optimizing?
                  pipe_params, # what parameters values are we searching?
                  cv=3,# 3-fold cross-validation.
                 n_jobs = -1)


Xvader = df[['body','vaderSentiment']]
Xvaderscore = df[['body','senti_score']]
Xbody = df[['body']]
y = df['is_2016']

model_vader = model_go(X=Xvader,
         y=y,
         gridsearch=gs)

model_vader_score = model_go(X=Xvaderscore,
        y=y,
        gridsearch=gs)

model_body = model_go(X=Xbody,
                y=y,
                gridsearch=gs)
