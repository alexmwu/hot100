#!/usr/bin/env python

# Scrapes the site http://billboardtop100of.com/ for all Billboard
# Hot 100 year end chart songs (e.g., all Hot 100 songs in the year
# end chart for 2014)


import urllib2
from datetime import date
from bs4 import BeautifulSoup

bbSite = 'http://billboardtop100of.com/'

startYear = 1941

currYear = date.today().year
# iterate to the current year - 1
for y in xrange(startYear, currYear):
    pass
y = startYear
response = urllib2.urlopen(bbSite + str(y) + '-2/')
html = response.read()
soup = BeautifulSoup(html, 'html.parser')
print soup.table
# 3 cases: no table, just list of hot 100 hits, table with
# cells with <a> tags, and cell without <a> tags

