#!/usr/bin/env python

# Check which lyrics are null

import pandas
import numpy as np

# user functions
from bagOfWords import getDF, split_tokenize

LYRICS_PATH = 'data/lyrics/'

### Processing training set ###
lyricsDF = getDF(LYRICS_PATH, train=True)

# Transform training data into feature vectors
lyricsDFNotNull = lyricsDF[pandas.notnull(lyricsDF['LYRICS'])]

print lyricsDF.shape
print lyricsDF.shape[0] - lyricsDFNotNull.shape[0]

