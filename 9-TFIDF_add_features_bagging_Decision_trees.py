#!/usr/bin/env python
# coding: utf-8

# # Modeling using TFIDF on `'body'`, additional features, Bagging with Decision Tree & GridSearch

# In[1]:


import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import stop_words
from sklearn.compose import ColumnTransformer
from tokenizer.tokenizer import RedditTokenizer
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score

import nltk
from nltk import word_tokenize  
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.stem.porter import PorterStemmer

import regex
import time

from my_tools.pipes_grids import *


# In[2]:


nltk_stops = set(stopwords.words('english'))


# In[3]:


df = pd.read_csv('datasets/data2.csv')
df.sample(2)


# In[4]:


features_txt = ['body']
features_non_txt = list(df.drop(columns=features_txt+['score','is_2016','senti_score']).columns.values)
features = features_txt + features_non_txt


# ## Define the pipeline and gridsearch here to be used for this round of analysis.

# In[5]:


## some implementation ideas from https://jorisvandenbossche.github.io/blog/2018/05/28/scikit-learn-columntransformer/
# ColumnTransformer(transformers,
#                   remainder=’drop’,
#                   sparse_threshold=0.3,
#                   n_jobs=-1,
#                   )

ct = ColumnTransformer(
    [
        ('tvec',TfidfVectorizer(), 'body'),
        #('ss',StandardScaler(),features_non_txt)
    ],
                  remainder='passthrough',
                  sparse_threshold=0.3,
                  n_jobs=-1,
                  )

pipe = Pipeline([
#     ('tvec',TfidfVectorizer()),
    ('ct',ct),
    ('bagclass',  BaggingClassifier(n_jobs=-1))
])

pipe_params = {
    'ct__tvec__tokenizer' : [RedditIt()],
    'ct__tvec__ngram_range' : [(1,1),(1,2)],
    'ct__tvec__max_features' : [None],
    'ct__tvec__min_df' : [.02,.05],
    'ct__tvec__max_df' : [.9,.85],
    'ct__tvec__stop_words' : [nltk_stops],
    'bagclass__n_estimators' : [10],

    

    
    
    
}


gs = GridSearchCV(pipe, # what object are we optimizing?
                  pipe_params, # what parameters values are we searching?
                  cv=3,# 3-fold cross-validation.
                 n_jobs = -1) 


# ## Define functions to automate the model fitting, gridsearch & score outputs
# see [`my_tools/pipes_grids.py`](my_tools/pipes_grids.py)

# ## Model Creation and Scoring

# ### All Comments and Submission

# In[ ]:





# In[6]:


sample = df[df['is_comment'] == 0]
X = df[['body','vaderSentiment']]
y = df['is_2016']

model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:


model[0].best_params_


# In[ ]:



X = sample[['body']]

model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:




