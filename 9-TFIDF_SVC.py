#!/usr/bin/env python
# coding: utf-8

# # Modeling using TFIDF on `'body'`, SVC & Gridsearch

# In[16]:


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


# In[2]:


nltk_stops = set(stopwords.words('english'))


# In[4]:


try:
    df = pd.read_csv('datasets/data_submissions.csv')
except:
    df = pd.read_csv('https://www.dropbox.com/s/eh48hdw4af7tjc3/data_submissions.csv?dl=0')


# In[5]:


features_txt = ['body']
features_non_txt = list(df.drop(columns=features_txt+['score','is_2016','senti_score']).columns.values)
features = features_txt + features_non_txt


# ## Define the pipeline and gridsearch here to be used for this round of analysis.

# In[24]:


ct = ColumnTransformer(
    [
        ('tvec',TfidfVectorizer(), 'body'),
    ],
                  remainder='passthrough',
                  sparse_threshold=0.3,

                  )



pipe = Pipeline([
    ('ct',ct),
    ('svc',SVC(gamma='scale'))

])

pipe_params = {
    'ct__tvec__tokenizer' : [None,RedditIt(),LemmaTokenizer()],
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
                 )


# ## Define functions to automate the model fitting, gridsearch & score outputs
# see [`my_tools/pipes_grids.py`](my_tools/pipes_grids.py)

# ## Model Creation and Scoring

# ### All Comments and Submission

# In[25]:


sample = df[df['is_comment'] == 0].sample(1000)
X = sample[['body','vaderSentiment']]
y = sample['is_2016']

model_vader = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[26]:


model_vader[0].best_params_


# In[27]:


X = sample[['body']]

model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[28]:


model[0].best_params_


# In[22]:


X = sample[['body','senti_score']]

model_vaderscore = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:


X = df['body']
y = df['is_2016']
model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:


X = df[['body','vaderSentiment']]
y = df['is_2016']
model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:


X = df[['body','senti_score']]
y = df['is_2016']
model = model_go(X=X,
        y=y,
        gridsearch=gs)


# In[ ]:
