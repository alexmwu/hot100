#!/usr/bin/env python

# Scrapes the site http://billboardtop100of.com/ for all Billboard
# Hot 100 year end chart songs (e.g., all Hot 100 songs in the year
# end chart for 2014)


import urllib2
from datetime import date

bbSite = 'http://http://billboardtop100of.com/'

startYear = 1941

currYear = date.today().year
# iterate to the current year - 1
for y in xrange(startYear, currYear):
    response = urllib2.urlopen(bbSite + y + '-2/')
    html = response.read()
