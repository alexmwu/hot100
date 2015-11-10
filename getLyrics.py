#!/usr/bin/env python

# grabs lyrics for all the songs in an input file
# file must be an '@' separated value (.atsv)
# this is due to the common occurrence of ','s in
# music titles and artist names
# There should be three columns of values

import requests
from bs4 import BeautifulSoup
import json

lyricsSite = 'http://www.lyrics.wikia.com/api.php'
params = {
        'action': 'lyrics',
        'artist': 'j cole',
        'song': 'no role modelz'
        }

# r = requests.get(lyricsSite, params = params)
r = requests.get(lyricsSite, params = params)

soup = BeautifulSoup(r.text, 'html.parser')
link = 0
anc = soup.a

if soup.a:
    if soup.a.has_attr('href'):
        link = anc['href']
print link
print soup.a
