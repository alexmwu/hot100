#!/usr/bin/env python

# Reads in lyrics from data/lyrics/####hot100.atsv files
# Creates bag of words from lyrics

# Referenced for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
import re
from nltk.corpus import stopwords # import stop word list
from sklearn.feature_extraction.text import CountVectorizer

BAGSIZE = 50

def cleanLyrics(raw_lyrics):
	# Remove non-letters
	letters_only = re.sub('[^a-zA-Z]', ' ', raw_lyrics)

	# Convert to lower case and split into individual words
	words = letters_only.lower().split()

	# Searching set is faster than searching list--convert to set
	stops = set(stopwords.words("english"))

	# Remove stop words
	meaningful_words = [w for w in words if not w in stops]

	# Join words back into string separated by space
	return (" ".join(meaningful_words))


train = pandas.read_csv('data/lyrics/1947hot100.atsv', \
	header=None, delimiter='@', na_filter=True, quoting=3, \
	names=['NUM', 'ARTIST', 'SONG', 'LYRICS'])
	# quoting=3 ignores double quotes

#train.shape
#print train["LYRICS"][0]
#print train["LYRICS"][1]

#lyrics = cleanLyrics(train['LYRICS'][0])
#print lyrics

# Clean lyrics
num_lyrics = train['LYRICS'].size
clean_lyrics = []
print "Cleaned %d of %d lyrics" % (0, num_lyrics)
for i in range(num_lyrics):
	if (i+1)%25 == 0:
		print "Cleaned %d of %d lyrics" % (i+1, num_lyrics)
	if not pandas.isnull(train['LYRICS'][i]):
		clean_lyrics.append(cleanLyrics(train['LYRICS'][i]))
print "Cleaned %d of %d lyrics" % (num_lyrics, num_lyrics)


# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = BAGSIZE) 
# Fit model and learn vocabulary
# Transform training data into feature vectors
train_data_features = vectorizer.fit_transform(clean_lyrics)
print train_data_features.shape

vocab = vectorizer.get_feature_names()
print vocab

# Print counts of each word in vocabulary
word_count = []
dist = train_data_features.toarray().sum(axis=0) # flattens matrix
for tag, count in zip(vocab, dist):
	#print count, tag
	word_count.append((count, tag))
word_count_sorted = sorted(word_count, key=lambda x: -x[0]) # decreasing
for count, word in word_count_sorted:
	print count, word






