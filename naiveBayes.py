#!/usr/bin/env python

# Multinomial Naive Bayes on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Reads in lyrics from data/lyrics/####hot100.atsv files
# Creates bag of words from lyrics
# Creates random forest model on training dataset
# Uses model to predict test set
# Prints accuracy of model on test set

# Reference for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import lda

# user functions
from bagOfWords import getDF, split_tokenize, testAccuracy

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

# Fit model and learn vocabulary on existing lyrics
# Transform training data into feature vectors
trainDFNotNull = trainDF[pandas.notnull(trainDF['LYRICS'])]
trainDataFeatures = vectorizer.fit_transform(trainDFNotNull['LYRICS'])
#printFeatures(vectorizer, trainDataFeatures)

### Multinomial Naive Bayes ###
# Initialize multinomial naive bayes model
mnb = MultinomialNB()

# Fit forest model to training set using bag of words as features and decade as label
mnb = mnb.fit(trainDataFeatures, trainDFNotNull['DECADE'])

# Get bag of words for test set, transform to feature vectors, and convert to numpy array
testDF = getDF(LYRICS_PATH_TEST, train=False)
testDFNotNull = testDF[pandas.notnull(testDF['LYRICS'])]
testDataFeatures = vectorizer.transform(testDFNotNull['LYRICS'])
testDataFeatures = testDataFeatures.toarray()

# Use random forest to make decade label predictions
result = mnb.predict(testDataFeatures)
# Copy results to pandas DF with predicted 'DECADE' column
output = pandas.DataFrame(data = {	\
	'NUM':testDFNotNull['NUM'],				\
	'ARTIST':testDFNotNull['ARTIST'],	\
	'SONG':testDFNotNull['SONG'],			\
	'YEAR':testDFNotNull['YEAR'],			\
	'DECADE':result})

# Write csv output file
output.to_csv(OUTPUT_PATH, index=False, sep='@', quoting=3, \
        columns=['NUM','ARIST', 'SONG', 'YEAR', 'DECADE'])

print 'Accuracy: ' + '%.2f' % testAccuracy(output) + '%'

