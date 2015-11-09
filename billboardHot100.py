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
# 3 cases: no table, just list of hot 100 hits, table with
# cells with <a> tags, and cell without <a> tags
if soup.table:
    rows = soup.table.findChildren(['tr'])
    for row in rows:
        cells = row.findChildren('td')
        for cell in cells:
# check if the td has an <a> tag
            links = cell.findChildren('a')
            if len(links) == 0:
                print cell
            else:
                for link in links:
                    print link
    # for tr in soup.table:
# # handle cells with <a> tags
        # links = soup.findChildren('a')
        # if len(links) == 0:
            # print tr
        # else:
            # for link in links:
                # print link
