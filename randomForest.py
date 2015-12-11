#!/usr/bin/env python

# Random Forest Classifer on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Creates random forest model on training dataset
# Uses model to predict test set
# Prints accuracy of model on test set

# Reference for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt

# user functions
from bagOfWords import testAccuracy

# run and import all variables in baseClassifier
from baseClassifier import *

CV = 10
OUTPUT_PATH = 'data/random_forest_classifier/output.csv'
CV_PLOT_PATH = 'data/random_forest_classifier/cross_validation.png'
TEST_PLOT_PATH = 'data/random_forest_classifier/test_prediction.png'

# Fit model and learn vocabulary on existing lyrics
### Random Forest Classifier ###
# Initialize random forest with 100 trees
forest = RandomForestClassifier(n_estimators=100)

# Cross validation on training data
train_predicted = cross_validation.cross_val_predict(forest, trainDataFeatures, y=trainDFNotNull['DECADE'], cv=CV)
score = metrics.accuracy_score(trainDFNotNull['DECADE'], train_predicted)
print 'Training Accuracy:', str(score*100) + '%'

fig, ax = plt.subplots()
ax.scatter(trainDFNotNull['YEAR'], train_predicted)
ax.plot([trainDFNotNull['DECADE'].min(), trainDFNotNull['DECADE'].max()], [trainDFNotNull['DECADE'].min(), trainDFNotNull['DECADE'].max()])
#ax.set_title(str(CV)+'-Fold Cross Validation')
ax.set_xlabel('Measured Year')
ax.set_ylabel('Predicted Decade')
plt.savefig(CV_PLOT_PATH, bbox_inches='tight')

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

print 'Testing Accuracy: ' + '%.2f' % testAccuracy(output) + '%'

ax.scatter(testDFNotNull['YEAR'], result)
ax.plot([testDFNotNull['DECADE'].min(), testDFNotNull['DECADE'].max()], [testDFNotNull['DECADE'].min(), testDFNotNull['DECADE'].max()])
#ax.set_title('Test Prediction')
ax.set_xlabel('Measured Year')
ax.set_ylabel('Predicted Decade')
plt.savefig(TEST_PLOT_PATH, bbox_inches='tight')
