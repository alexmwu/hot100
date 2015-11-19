#!/usr/bin/env python

# grabs lyrics for all the songs in an input file
# file must be an '@' separated value (.atsv)
# this is due to the common occurrence of ','s in
# music titles and artist names
# There should be three columns of values

# Writes lyric file separated by @: song number@artist@song title@lyrics

import sys
import requests
from bs4 import BeautifulSoup, Comment
import json
import types
from os import listdir
from os.path import isfile, join, basename
from unidecode import unidecode

LYRICSSITE = 'http://www.lyrics.wikia.com/api.php'
CHARTS_PATH = sys.argv[1]
LYRICS_PATH = 'data/lyrics/'
NUMBER, ARTIST, SONG = range(3)

# adapted from http://stackoverflow.com/questions/10491223/how-can-i-turn-br-and-p-into-line-breaks
def replace_with_newlines(elem):
    text = ''
    for elem in elem.descendants:
        if isinstance(elem, types.StringTypes):
            text += unidecode(elem.strip())
        elif elem.name == 'br':
            text += ' '
        #    text += '\n'
    return text

filename = basename(CHARTS_PATH)
f = open(CHARTS_PATH)
print filename
lyrics_file = open(LYRICS_PATH+filename, 'w') # truncates existing file
songs = f.readlines()
f.close()
for song in songs:
    song_params = song.split('@')
    if len(song_params) != 3:
        print 'Error with', filename
        break
    params = {
        'action': 'lyrics',
        'artist': song_params[ARTIST],
        'song': song_params[SONG]
    }
    # r = requests.get(lyricsSite, params = params)
    link_r = requests.get(LYRICSSITE, params = params)

    link_soup = BeautifulSoup(link_r.text, 'html.parser')
    link = 0
    anc = link_soup.a

    if link_soup.a:
        if link_soup.a.has_attr('href'):
            link = anc['href']
    if link != 0:
        lyrics_r = requests.get(link)

        lyrics_soup = BeautifulSoup(lyrics_r.text, 'html.parser')

        comments = lyrics_soup.findAll(text = lambda text: isinstance(text, Comment))
        # extract comments
        [c.extract() for c in comments]

        # extract <script> tags
        [s.extract() for s in lyrics_soup('script')]

        lyrics_divs = lyrics_soup.findAll('div', { 'class': 'lyricbox'})
        line = song.rstrip('\n') + '@'
        for div in lyrics_divs:
            line += replace_with_newlines(div)
        line += '\n'
        lyrics_file.write(line)    
lyrics_file.close()
print 'Finished processing', filename

'''
params = {
        'action': 'lyrics',
        'artist': 'j cole',
        'song': 'no role modelz'
        }

'''