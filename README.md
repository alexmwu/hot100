Dependencies:
=============

Beautiful Soup 4
----------------
`pip install beautifulsoup4`

unidecode
---------
`pip install unidecode`

requests
--------
`pip install requests`

ntlk
----
`pip install -U nltk`
To install required packages, open python (i.e., by typing `python` in the command line) and
do the following
```
import nltk
nltk.download()
```
In the window that pops up, download `all-corpora`

scikit-learn
------------
`pip install -U scikit-learn` or `conda install scikit-learn`

scikit-learn requires scipy
`pip install scipy`

numpy
-----
`pip install numpy`

lda
---
`pip install lda`

matplotlib
----------
`pip install matplotlib`

LyricWiki API
-------------
This is subject to change.

References:
=============
All scikit-learn model documentation can be found at http://scikit-learn.org/stable/index.html

https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

https://pypi.python.org/pypi/lda

Notes:
======
Charts format: NUM@ARTIST@SONG

Lyrics format: NUM@ARTIST@SONG@LYRICS

Removes possessives ('s), trailing apostrophes, and concatenates words divided by hyphens


Files:
======
accuracy.py: reads output file containing predicted classes from test dataset and computes the accuracy of the model prediction

bagOfWords.py: several functions are implemented that can be useful for preprocessing and manipulating a bag of words using scikit learn model

baseClassifiers.py: base script on which other classifiers are built

billboardHot100.py: scrapes Billboard Hot 100 for top rated songs in the past 80 years and writes to data/charts

checkEmpty.sh: checks which charts are empty (see which years billboardHot100 failed on)

clustering.py: implements k-means clustering (k = 7) on the text

getLyrics.py: processes all songs listed in every file in data/charts and writes lyrics scraped from http://www.lyrics.wikia.com/api.php to `data/lyrics`

getLyricsParallel.py: processes one charts file and writes lyrics scraped from http://www.lyrics.wikia.com/api.php to `data/lyrics`. Can be called in parallel to process multiple charts files.

naiveBayes.py: creates model for Multinomial Naive Bayes from training dataset to predict on test dataset. Also outputs predictions to csv file in `data/Bag_of_words_model.csv`.

randomForest.py: creates model for Random Forest Classifier from training dataset to predict on test dataset. Also outputs predictions to csv file in `data/Bag_of_words_model.csv`.

rng.sh: randomly selects files to be training files and testing files

runParallel.sh: bash script to call getLyricsParallel.py in parallel with different files within data/charts as parameters.

topicModeling.py: runs topic modeling (lda) on all lyrics

