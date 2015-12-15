#!/usr/bin/env python

# Topic Modeling (latent dirichlet allocation) on Hot100 song lyrics to predict decade song was released depending on vocabulary in lyrics

# Grabs all lyrics
# Runs topic modeling on all the lyrics
# Prints top 8 words in each topic

# https://pypi.python.org/pypi/lda

import pandas
#import lda
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

# user functions
from bagOfWords import getData, split_tokenize, printFeatures, create_count_top

LYRICS_PATH = 'data/lyrics/'
OUTPUT_PATH = 'data/topic_model_output.txt'

### Processing training set ###
lyricsDF = getData(LYRICS_PATH)

# Initialize the "CountVectorizer" object, which is scikit-learn's bag of words tool.
vectorizer = CountVectorizer(analyzer = 'word',   \
                             tokenizer = split_tokenize,    \
                             preprocessor = None, \
                             stop_words = None)
                             #max_features = 50)

# Transform training data into feature vectors
lyricsDFNotNull = lyricsDF[pandas.notnull(lyricsDF['LYRICS'])]
lyricsFeatures = vectorizer.fit_transform(lyricsDFNotNull['LYRICS'])

# Creates a list of lists to store number of top words in song by rank
# Also keeps track of the decade of the songs
list_song, decade_song = create_count_top(vectorizer, lyricsDF)


plt.figure(1)
x = []
y = []
for a in range(1,100):
	for b in list_song[a]:
		x.append(a) # Rank of song
		y.append(b) # Number of top words in song


# Different colors for each decade for the scatter plot		
c = []
for a in range(1,100):
	for b in decade_song[a]:
		if b == 1940:
			c.append('b')
		elif b == 1950:
			c.append('g')
		elif b == 1960:
			c.append('r')
		elif b == 1970:
			c.append('c')
		elif b == 1980:
			c.append('m')
		elif b == 1990:
			c.append('y')
		elif b == 2000:
			c.append('k')
		elif b == 2010:
			c.append('w')


#Print the scatter plot
plt.scatter(x,y, c=c)
plt.title("Occurance of top words and ranking")
plt.xlabel("Rank of Song")
plt.ylabel("# of Top 100 Words in Song")
plt.xlim(0, 101)
plt.ylim(0,80)
plt.plot(x, np.poly1d(np.polyfit(x, y, 1))(x))
plt.show()




