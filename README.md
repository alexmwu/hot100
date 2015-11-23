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

LyricWiki API
-------------
This is subject to change.

References:
-----------
https://www.kaggle.com/c/word2vec-nlp-tutorial/details/part-1-for-beginners-bag-of-words

Notes:
------
Charts format: NUM@ARTIST@SONG

Lyrics format: NUM@ARTIST@SONG@LYRICS

Splits contractions into two words