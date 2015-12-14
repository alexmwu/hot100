#!/usr/bin/env python

# Topic Modeling (latent dirichlet allocation) on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Grabs all lyrics
# Runs topic modeling on all the lyrics
# Prints top 8 words in each topic

# https://pypi.python.org/pypi/lda

import pandas
#import lda
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# user functions
from bagOfWords import getDF, split_tokenize

LYRICS_PATH = 'data/lyrics/'
OUTPUT_PATH = 'data/topic_model_output.txt'

### Processing training set ###
lyricsDF = getData(LYRICS_PATH, train=True)

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = split_tokenize,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             )
                             #max_features = BAGSIZE)

# Transform training data into feature vectors
lyricsDFNotNull = lyricsDF[pandas.notnull(lyricsDF['LYRICS'])]
lyricsFeatures = vectorizer.fit_transform(lyricsDFNotNull['LYRICS'])

# Create lda topic modeler (20 topics, 300 iterations)
model = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

# Fit lda model on features
model.fit(lyricsFeatures)
topic_word = model.topic_word_
n_top_words = 20