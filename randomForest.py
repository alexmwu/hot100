#!/usr/bin/env python

# loads bag of words models and corresponding data frames to build random forest model on lyrics

import pandas
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from os.path import isfile, join, basename
from os import listdir
import sys
import numpy as np
import lda

def testAccuracy(result):
	nIncorrect = 0.0
	nSamples = result['DECADE'].size
        # indices may not be in order or all present because of null checks
        # so, don't use range (just iterate over the iterable index of the df)
	for i in result.index:
		if pandas.isnull(result['DECADE'][i]):
			nIncorrect += 1.0
		elif int(result['YEAR'][i]//10*10) != result['DECADE'][i]:
			nIncorrect += 1.0
	print nIncorrect, nSamples
	return nIncorrect/nSamples

# Usage: trainAvgFeatureVec = createAvgFeatureVec(vectorizer, trainDataFeatures)
def createAvgFeatureVec(vectorizer, features):
	# Useful for Naive Bayes, not used for Random Forest
	nwords = features.toarray().sum(axis=0).sum() # flattens matrix to single sum
	avgFeatureVec = []
	for feature in features:
		avgFeatureVec.append(feature/nwords)
	return avgFeatureVec

'''
# Create lda topic modeler (20 topics, 1500 iterations)
model = lda.LDA(n_topics=20, n_iter=300, random_state=1)

# Fit lda model on features
model.fit(trainDataFeatures)
topic_word = model.topic_word_
n_top_words = 8

# Iterate through topics (i: topic number, topic_dist: distribution of items in topic)
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
'''

### Random Forest Classifier ###
# Initialize random forest with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Fit forest model to training set using bag of words as features and decade as label
forest = forest.fit(trainDataFeatures, trainDFNotNull['DECADE'])

# Get bag of words for test set, transform to feature vectors, and convert to numpy array
testDF = getDF(LYRICS_PATH_TEST, train=False)
testDFNotNull = testDF[pandas.notnull(testDF['LYRICS'])]
testDataFeatures = vectorizer.transform(testDFNotNull['LYRICS'])
testDataFeatures = testDataFeatures.toarray()

# Use random forest to make decade label predictions
result = forest.predict(testDataFeatures)
# Copy results to pandas DF with predicted 'DECADE' column
# Write csv output file
output = pandas.DataFrame(data = {	\
	'NUM':testDFNotNull['NUM'],				\
	'ARTIST':testDFNotNull['ARTIST'],	\
	'SONG':testDFNotNull['SONG'],			\
	'YEAR':testDFNotNull['YEAR'],			\
	'DECADE':result})

print 'accuracy: ', str(1 - testAccuracy(output))

