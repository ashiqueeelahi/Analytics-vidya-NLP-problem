

Let us begin with importing the libraries and datasets

import pandas as pd
train = pd.read_csv('../input/analytics-vidya-nlp-data/train_E6oV3lV.csv');
test = pd.read_csv('../input/analytics-vidya-nlp-data/test_tweets_anuFYb8.csv');
ss = pd.read_csv('../input/analytics-vidya-nlp-data/sample_submission_gfvA5FD.csv')

import numpy as np
import pandas as pd
import re
import nltk
import spacy
import string

LOWERING

train["text_lower"] = train["tweet"].str.lower();
test["text_lower"] = test["tweet"].str.lower()
train.head()


REMOVING PUNCTUATION

PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text):
    """custom function to remove the punctuation"""
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

train["text_wo_punct"] = train["text_lower"].apply(lambda text: remove_punctuation(text));
test["text_wo_punct"] = test["text_lower"].apply(lambda text: remove_punctuation(text))
train.head()


STOPWORDS

from nltk.corpus import stopwords
", ".join(stopwords.words('english'))



STOPWORDS = set(stopwords.words('english'))
def remove_stopwords(text):
    """custom function to remove the stopwords"""
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

train["text_wo_stop"] = train["text_wo_punct"].apply(lambda text: remove_stopwords(text))
test["text_wo_stop"] = test["text_wo_punct"].apply(lambda text: remove_stopwords(text))
train.head()


tr = train.drop(columns = ['tweet', 'text_lower', 'text_wo_stop'], axis = 1);
te = test.drop(columns = ['tweet', 'text_lower', 'text_wo_stop'], axis = 1);

STEMMING

from nltk.stem.porter import PorterStemmer


stemmer = PorterStemmer()
def stem_words(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

tr["text_stemmed"] = tr["text_wo_punct"].apply(lambda text: stem_words(text));
te["text_stemmed"] = te["text_wo_punct"].apply(lambda text: stem_words(text))
tr.head()


LEMMATIZING

from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
def lemmatize_words(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

tr["text_lemmatized"] = tr["text_stemmed"].apply(lambda text: lemmatize_words(text));
te["text_lemmatized"] = te["text_stemmed"].apply(lambda text: lemmatize_words(text))
tr.head()



import random
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import xgboost as xgb;
from xgboost import XGBClassifier;
from xgboost import XGBRegressor;

LABEL ENCODING AND SPLITING INTO TEST AND TRAIN DATA

x = tr.text_lemmatized.values;
y = tr.label.values

lab = LabelEncoder();
x = lab.fit_transform(x)

x



x = x.reshape(-1, 1)

y



z = te.text_lemmatized.values

z = lab.fit_transform(z)

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 55)

xtrain



ytrain


RANDOMFORREST

rfc = RandomForestClassifier();
rfc.fit(xtrain, ytrain);
rfc.score(xtest, ytest)

0.9042702956358517

XGBOOST

xx = XGBClassifier();
xx.fit(xtrain, ytrain);
xx.score(xtest, ytest)

0.9355545127483185

LIGHTGBM

from lightgbm import LGBMClassifier
lgb = LGBMClassifier(n_estimators=10)
lgb.fit(xtrain,ytrain);
lgb.score(xtest, ytest)

0.9282027217268888

KNEARESTNEIGHBOUR

knn = KNeighborsClassifier();
knn.fit(xtrain, ytrain);
knn.score(xtest, ytest)

0.9352416705771938

DECISION TREE

dc = DecisionTreeClassifier();
dc.fit(xtrain, ytrain);
dc.score(xtest, ytest)

0.9042702956358517

We are getting best results from xgboost.Its better and safer to go with that one I guess.
