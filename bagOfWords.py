#!/usr/bin/env python

# Functions to read in lyrics from data/lyrics/####hot100.atsv files
# and create bag of words from lyrics

# Referenced for bag of words: https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

import pandas
import re
import math
import itertools
from nltk.corpus import stopwords # import stop word list
from sklearn.feature_extraction.text import CountVectorizer
from os.path import isfile, join, basename
from os import listdir, stat
from collections import Counter

# Searching set is faster than searching list--convert to set
stops = set(stopwords.words("english"))

BAGSIZE = 50

# Custom tokenizer for scikit CountVectorizer because it would strip apostrophe's
def split_tokenize(s):
	return s.split()

# Usage: cleanLyrics(raw_lyrics)
# Input: string
# Output: lower case string with removed special characters except for apostrophes within a word (removes trailing apostrophes)
def cleanLyrics(raw_lyrics):
	# Remove non-letters, convert to lowercase, remove stop words
	letters_only = re.sub("[^-'a-zA-Z]", ' ', raw_lyrics)
	letters_only = re.sub("(-|'s)", '', letters_only)
	words = letters_only.lower().split()
	meaningful_words = [w.strip('\'') for w in words if not w in stops]
	return (" ".join(meaningful_words))

# Usage: printFeatures(vectorizer, trainDataFeatures)
def printFeatures(vectorizer, features):
	# Print counts of each word in vocabulary in sorted descending order
	vocab = vectorizer.get_feature_names()
	word_count = []
	dist = features.toarray().sum(axis=0) # flattens matrix
	for tag, count in zip(vocab, dist):
		word_count.append((count, tag))
	word_count_sorted = sorted(word_count, key=lambda x: -x[0]) # decreasing
	for count, word in word_count_sorted:
		print count, word

# Usage: trainAvgFeatureVec = createAvgFeatureVec(vectorizer, trainDataFeatures)
def createAvgFeatureVec(vectorizer, features):
	# Useful for Naive Bayes, not used for Random Forest
	nwords = features.toarray().sum(axis=0).sum() # flattens matrix to single sum
	avgFeatureVec = []
	for feature in features:
		avgFeatureVec.append(feature/nwords)
	return avgFeatureVec

# Usage: getDF(PATH, train)
# Input: PATH = directory path that contains lyrics
# Input: train: True or False; True denotes training dataset, False denotes testing dataset
# Creates data frame that stores Hot100 number ranking for given year, Artist, Song, Lyrics, Year, Decade
# Decade column is left Null for testing dataset
def getDF(LYRICS_PATH, train):
	columns = ['NUM', 'ARTIST', 'SONG', 'LYRICS', 'YEAR', 'DECADE']
	df = pandas.DataFrame(columns=columns)

	# Ignores hidden files
	files = [f for f in listdir(LYRICS_PATH) if isfile(join(LYRICS_PATH,f)) and not f.startswith('.')]
	for FILE in files:
                if stat(LYRICS_PATH+FILE).st_size <= 0:
                    continue
		df_singleFile = pandas.read_csv(LYRICS_PATH+FILE, \
			header=None, delimiter='@', na_filter=True, quoting=3, \
			names=['NUM', 'ARTIST', 'SONG', 'LYRICS'])
			# quoting=3 ignores double quotes

		# Adding YEAR column to data frame
		fileYear = int(FILE[:-len('hot100.atsv')])
		yearCol = pandas.DataFrame({'YEAR':[fileYear]*df_singleFile.shape[0]})
		df1 = pandas.concat([df_singleFile, yearCol], axis=1)

		# Add DECADE column to data frame for training set
		if train:
			fileDecade = fileYear//10*10
			decadeCol = pandas.DataFrame({'DECADE':[fileDecade]*df_singleFile.shape[0]})
			df1 = pandas.concat([df1, decadeCol], axis=1)

		# Clean lyrics in place
		num_lyrics = df1['LYRICS'].size
		# print "Cleaned %d of %d lyrics" % (0, num_lyrics)
		for i in range(num_lyrics):
			#if (i+1)%25 == 0:
			#	print "Cleaned %d of %d lyrics" % (i+1, num_lyrics)
			if not pandas.isnull(df1['LYRICS'][i]):
				df1.loc[i,'LYRICS'] = cleanLyrics(df1['LYRICS'][i])
		#print "Finished cleaning %d lyrics" % (num_lyrics)

		# Append new DF to master DF
		df = df.append(df1, ignore_index=True)
	return df

# Usage: testAccuracy(result)
# Input: result = testing dataset predicted on model in dataframe format
# Output: Returns float, accuracy % of model on testing data
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
	return (1-nIncorrect/nSamples)*100

# Usage: list_song, decade_song = create_count_top(vectorizer, lyricsDF)
def create_count_top(vectorizer, features):

	# Get the list of all words in a song
	avgFeatureVec = []
	for feature in features.values:
		words = feature[3]
		decade = feature[5]
		if not isinstance(words,float):
			word_list = words.split(' ')
			avgFeatureVec.append(word_list)

	# Get a count of the most common words in all the song lyrics
	merged = list(itertools.chain(*avgFeatureVec))
	count = Counter()
	for word in merged:
		count[word] += 1
	top_100 =  count.most_common(100)

	list_song = [[0 for x in range(101)] for x in range(101)] 
	decade_song = [[0 for x in range(101)] for x in range(101)] 
	i = 0

	# Group together in list the number of top words in a song based on a song's rank
	for feature in features.values:
		words = feature[3]
		rank = int(feature[0])
		decade = int(feature[5])
		num = 0
		if not isinstance(words,float):
			for word in top_100:
				if word[0] in words:
					num = num + 1
			list_song[rank][i] = num # Number of top words
			decade_song[rank][i] = decade # Decade to map to song

			if i == 100: i = 0
			else: i = i + 1
			
			print "Rank: " + str(feature[0]) + " song: " + feature[2] + " has " + str(num) + " top words"
	return list_song, decade_song

# To get all the songs
def getData(LYRICS_PATH):
	columns = ['NUM', 'ARTIST', 'SONG', 'LYRICS', 'YEAR', 'DECADE']
	df = pandas.DataFrame(columns=columns)

	# Ignores hidden files
	files = [f for f in listdir(LYRICS_PATH) if isfile(join(LYRICS_PATH,f)) and not f.startswith('.')]
	for FILE in files:
                if stat(LYRICS_PATH+FILE).st_size <= 0:
                    continue
		df_singleFile = pandas.read_csv(LYRICS_PATH+FILE, \
			header=None, delimiter='@', na_filter=True, quoting=3, \
			names=['NUM', 'ARTIST', 'SONG', 'LYRICS'])
			# quoting=3 ignores double quotes

		# Adding YEAR column to data frame
		fileYear = int(FILE[:-len('hot100.atsv')])
		yearCol = pandas.DataFrame({'YEAR':[fileYear]*df_singleFile.shape[0]})
		df1 = pandas.concat([df_singleFile, yearCol], axis=1)

		# Add DECADE column to data frame for training set
		fileDecade = fileYear//10*10
		decadeCol = pandas.DataFrame({'DECADE':[fileDecade]*df_singleFile.shape[0]})
		df1 = pandas.concat([df1, decadeCol], axis=1)

		# Clean lyrics in place
		num_lyrics = df1['LYRICS'].size
		# print "Cleaned %d of %d lyrics" % (0, num_lyrics)
		for i in range(num_lyrics):
			#if (i+1)%25 == 0:
			#	print "Cleaned %d of %d lyrics" % (i+1, num_lyrics)
			if not pandas.isnull(df1['LYRICS'][i]):
				df1.loc[i,'LYRICS'] = cleanLyrics(df1['LYRICS'][i])
		#print "Finished cleaning %d lyrics" % (num_lyrics)

		# Append new DF to master DF
		df = df.append(df1, ignore_index=True)
	return df

