#!/usr/bin/env python

# Base Classifer Code on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics
# imported from other learner scripts

# Reads in lyrics from data/lyrics/####hot100.atsv files
# Creates bag of words from lyrics

# Reference for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import cross_validation
import numpy as np

# user functions
from bagOfWords import getDF, split_tokenize

BAGSIZE = 100
LYRICS_PATH = 'data/lyrics/'

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = split_tokenize,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             )
                             #max_features = BAGSIZE)

# Transform data into feature vectors
dataDF = getDF(LYRICS_PATH, train=True)
dataDFNotNull = dataDF[pandas.notnull(dataDF['LYRICS'])]

# Split dataset to train and test with a 4:1 ratio
trainDFNotNull, testDFNotNull = cross_validation.train_test_split(dataDFNotNull, test_size=0.2)
trainDataFeatures = vectorizer.fit_transform(trainDFNotNull['LYRICS'])
testDataFeatures = vectorizer.transform(testDFNotNull['LYRICS'])
testDataFeatures = testDataFeatures.toarray()