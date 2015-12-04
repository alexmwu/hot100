#!/usr/bin/env python

# Random Forest Classifer on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Creates random forest model on training dataset
# Uses model to predict test set
# Prints accuracy of model on test set

# Reference for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
from sklearn.ensemble import RandomForestClassifier

# user functions
from bagOfWords import testAccuracy
# run and import all variables in baseClassifier
from baseClassifier import *

# Fit model and learn vocabulary on existing lyrics
### Random Forest Classifier ###
# Initialize random forest with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit forest model to training set using bag of words as features and decade as label
forest = forest.fit(trainDataFeatures, trainDFNotNull['DECADE'])

# Use random forest to make decade label predictions
result = forest.predict(testDataFeatures)
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

