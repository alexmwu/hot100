#!/usr/bin/env python

# Reads in lyrics from data/lyrics/####hot100.atsv files
# Creates bag of words from lyrics

# Referenced for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
import re
from nltk.corpus import stopwords # import stop word list

def cleanLyrics(raw_lyrics):
	# Remove non-letters
	letters_only = re.sub("[^a-zA-Z]", " ", raw_lyrics)

	# Convert to lower case and split into individual words
	words = letters_only.lower().split()

	# Searching set is faster than searching list--convert to set
	stops = set(stopwords.words("english"))

	# Remove stop words
	meaningful_words = [w for w in words if not w in stops]

	# Join words back into string separated by space
	return (" ".join(meaningful_words))


train = pandas.read_csv("data/lyrics3/1947hot100.atsv", \
	header=None, delimiter="@", na_filter=True, quoting=3, \
	names=['NUM', 'ARTIST', 'SONG', 'LYRICS'])
# quoting=3 ignores double quotes

#train.shape
#print train["LYRICS"][0]
#print train["LYRICS"][1]

lyrics = cleanLyrics(train["LYRICS"][0])
print lyrics

# Clean lyrics
num_lyrics = train["LYRICS"].size
clean_lyrics = []
for i in range(num_lyrics):
	if (i+1)%25 == 0:
		print "Cleaned %d of %d lyrics\n" % (i+1, num_lyrics)
	clean_lyrics.append(cleanLyrics(train["LYRICS"][i]))