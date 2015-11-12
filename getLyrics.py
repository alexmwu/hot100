#!/usr/bin/env python

# grabs lyrics for all the songs in an input file
# file must be an '@' separated value (.atsv)
# this is due to the common occurrence of ','s in
# music titles and artist names
# There should be three columns of values

import requests
from bs4 import BeautifulSoup, Comment
import json
import types

# adapted from http://stackoverflow.com/questions/10491223/how-can-i-turn-br-and-p-into-line-breaks
def replace_with_newlines(elem):
    text = ''
    for elem in elem.descendants:
        if isinstance(elem, types.StringTypes):
            text += elem.strip()
        elif elem.name == 'br':
            text += '\n'
    return text


lyricsSite = 'http://www.lyrics.wikia.com/api.php'
params = {
        'action': 'lyrics',
        'artist': 'j cole',
        'song': 'no role modelz'
        }

# r = requests.get(lyricsSite, params = params)
link_r = requests.get(lyricsSite, params = params)

link_soup = BeautifulSoup(link_r.text, 'html.parser')
link = 0
anc = link_soup.a

if link_soup.a:
    if link_soup.a.has_attr('href'):
        link = anc['href']

lyrics_r = requests.get(link)

lyrics_soup = BeautifulSoup(lyrics_r.text, 'html.parser')

comments = lyrics_soup.findAll(text = lambda text: isinstance(text, Comment))
# extract comments
[c.extract() for c in comments]

# extract <script> tags
[s.extract() for s in lyrics_soup('script')]

lyrics_divs = lyrics_soup.findAll('div', { 'class': 'lyricbox'})
for div in lyrics_divs:
    print replace_with_newlines(div)
