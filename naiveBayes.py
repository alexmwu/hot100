#!/usr/bin/env python

# Multinomial Naive Bayes on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Creates multinomial naive bayes model on training dataset
# Uses model to predict test set
# Prints accuracy of model on test set

import pandas
from sklearn.naive_bayes import MultinomialNB

# user functions
from bagOfWords import testAccuracy
# run and import all variables in baseClassifier
from baseClassifier import *

# Fit model and learn vocabulary on existing lyrics
### Multinomial Naive Bayes ###
# Initialize multinomial naive bayes model
mnb = MultinomialNB()

# Fit forest model to training set using bag of words as features and decade as label
mnb = mnb.fit(trainDataFeatures, trainDFNotNull['DECADE'])

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

