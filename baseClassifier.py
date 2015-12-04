#!/usr/bin/env python

# Base Classifer Code on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics
# imported from other learner scripts

# Reads in lyrics from data/lyrics/####hot100.atsv files
# Creates bag of words from lyrics

# Reference for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# user functions
from bagOfWords import getDF, split_tokenize

LYRICS_PATH_TRAIN = 'data/sample_lyrics_train/'
LYRICS_PATH_TEST = 'data/sample_lyrics_test/'
OUTPUT_PATH = 'data/Bag_of_Words_model.csv'

### Processing training set ###
trainDF = getDF(LYRICS_PATH_TRAIN, train=True)

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = split_tokenize,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             )
                             #max_features = BAGSIZE)

# Transform training data into feature vectors
trainDFNotNull = trainDF[pandas.notnull(trainDF['LYRICS'])]
trainDataFeatures = vectorizer.fit_transform(trainDFNotNull['LYRICS'])
#printFeatures(vectorizer, trainDataFeatures)

# Get bag of words for test set, transform to feature vectors, and convert to numpy array
testDF = getDF(LYRICS_PATH_TEST, train=False)
testDFNotNull = testDF[pandas.notnull(testDF['LYRICS'])]
testDataFeatures = vectorizer.transform(testDFNotNull['LYRICS'])
testDataFeatures = testDataFeatures.toarray()

