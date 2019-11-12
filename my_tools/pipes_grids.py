
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix, make_scorer, roc_auc_score, roc_curve


import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction import stop_words
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk import word_tokenize
import time

# Import CountVectorizer and TFIDFVectorizer from feature_extraction.text.
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from tokenizer.tokenizer import RedditTokenizer

#tokenizer

class PorterStemmerTokenizer(object):
    def __init__(self):
        self.wnl = PorterStemmer()
    def __call__(self, articles):
        return [self.wnl.stem(t) for t in word_tokenize(articles)]


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles)]


#preprocessor
class RedditIt(object):
    def __init__(self):
        self.wnl = RedditTokenizer()
    def __call__(self, articles):
        return [" ".join(self.wnl.tokenize(t)) for t in articles]





def model_go(X,y,gridsearch):
    t0 = time.time()
    # Redefine training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.33,
                                                        stratify=y)
    # Fit GridSearch to training data.
    gridsearch.fit(X_train,y_train)
    scores(gridsearch, X_train, X_test, y_train, y_test)
    return(gridsearch, [X_train, X_test, y_train, y_test])


def scores(model, X_train, X_test, y_train, y_test):
    # What's the best score?
    print(f'Best score: {model.best_score_}')
    # Score model on training set.
    print(f'Training score: {model.best_estimator_.score(X_train, y_train)}')
    # Score model on testing set.
    print(f'Test score: {model.best_estimator_.score(X_test, y_test)}')
    print(f'Cross val score: {cross_val_score(model.best_estimator_, X_train, y_train, cv=5).mean()}')
    print(f'Baseline Accuracy: {y_test.value_counts(normalize=True)[1]}')

    #thanks J Allan Hall!
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    print(f'Accuracy: {(tp + tn) / (tp+tn+fp+fn)}')
    print(f'Misclassification rate: {1 - ((tp + tn) / (tp+tn+fp+fn))}')
    print(f'Sensitivity & Recall: {tp / (tp + fn)}')
    print(f'Specificity: {tn / (tn+fp)}')
    print(f'Precision: {tp / (tp+fp)}')
    print(f'Balanced Accuracy Score: {balanced_accuracy_score(y_test, y_pred, adjusted=True)}')
