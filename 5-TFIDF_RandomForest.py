#!/usr/bin/env python
# coding: utf-8

# # Modeling using TFIDFVectorizer and RandomForestClassifier & GridSearch
# For each combination of transformers and predictors I am exploring different combinations of data.
# - All Comments & Submissions
# - Comments & Submission with Reddit Scores >10
# - Comments only
# - Submissions only

# In[1]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words

import nltk
from nltk.corpus import stopwords

from my_tools.pipes_grids import *


# In[2]:


sk_stop = set(stop_words.ENGLISH_STOP_WORDS)
nltk_stops = set(stopwords.words('english'))


# In[3]:


try:
    df = pd.read_csv('datasets/data_submissions.csv')
except:
    df = pd.read_csv('https://www.dropbox.com/s/eh48hdw4af7tjc3/data_submissions.csv?dl=0')


# ## Define the pipeline and gridsearch here to be used for this round of analysis.

# In[4]:


pipe = Pipeline([
    ('tvec', TfidfVectorizer()),
    ('rfc', RandomForestClassifier(n_jobs=-1))
])

pipe_params = {
    'tvec__ngram_range' : [(1,1),(1,2)],
    'tvec__max_features' : [1100,1500,1750],
    'tvec__min_df' : [.01, .025],
    'tvec__max_df' : [.8,.9],
    'tvec__stop_words' : [nltk_stops],
    'rfc__n_estimators' : [10],
    'rfc__min_samples_split' : [2,3],
    'rfc__min_samples_leaf' : [2,3,4],
    'rfc__max_features': ['sqrt','log2'],
    
}

gs = GridSearchCV(pipe, # what object are we optimizing?
                  pipe_params, # what parameters values are we searching?
                  cv=3,# 3-fold cross-validation.
                 n_jobs = -1) 


# ## Define functions to automate the model fitting, gridsearch & score outputs
# see [`my_tools/pipes_grids.py`](my_tools/pipes_grids.py)

# ## Model Creation and Scoring

# In[5]:


X = df['body']
y = df['is_2016']
model_base = model_go(X=X,
        y=y,
      gridsearch=gs)
model_base[0].best_params_


# In[ ]:


X = df[['body','vaderSentiment']]
y = df['is_2016']
model_vader = model_go(X=X,
        y=y,
        gridsearch=gs)
model_vader[0].best_params_


# In[ ]:


X = df[['body','senti_score']]
y = df['is_2016']
model_sentiscore = model_go(X=X,
        y=y,
        gridsearch=gs)
model_sentiscore[0].best_params_

