#!/usr/bin/env python

# Topic Modeling (latent dirichlet allocation) on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Grabs all lyrics
# Runs topic modeling on all the lyrics
# Prints top 8 words in each topic

# https://pypi.python.org/pypi/lda

import pandas
import lda
from sklearn.feature_extraction.text import CountVectorizer

# user functions
from bagOfWords import getDF, split_tokenize

LYRICS_PATH = 'data/lyrics/'
OUTPUT_PATH = 'data/topic_model_output.txt'

### Processing training set ###
lyricsDF = getDF(LYRICS_PATH, train=True)

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
model = lda.LDA(n_topics=20, n_iter=300, random_state=1)

# Fit lda model on features
model.fit(lyricsFeatures)
topic_word = model.topic_word_
n_top_words = 8

f = open(OUTPUT_PATH, 'w')

# Iterate through topics (i: topic number, topic_dist: distribution of items in topic)
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vectorizer.get_feature_names())[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    # write top #n_top_words to file
    out = 'Topic {}: {}'.format(i, ' '.join(topic_words))
    f.write(out)

f.close()

