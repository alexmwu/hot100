#!/usr/bin/env python

# Scrapes the site http://billboardtop100of.com/ for all Billboard
# Hot 100 year end chart songs (e.g., all Hot 100 songs in the year
# end chart for 2014)


import urllib2
from datetime import date
from bs4 import BeautifulSoup
from unidecode import unidecode
from os.path import isfile

bbSite = 'http://billboardtop100of.com/'

# well formatted tables start at 1945; 1959 and 2013 also have issues
# Additionally, 1940 has a resource id of 336 for whatever reason
start_year = 1941

curr_year = date.today().year
# iterate to the current year - 1
for y in xrange(start_year, curr_year):
    fname = 'data/charts/' + str(y) + 'hot100.atsv'
    if isfile(fname):
        continue
    fp = open(fname, 'w')

# 1959 has tables, but they are poorly formatted
# still create the file for it
    if y == 1959:
        fp.close()
        continue

    response = urllib2.urlopen(bbSite + str(y) + '-2/')
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser')
# 3 cases: no table, just list of hot 100 hits, table with
# cells with <a> tags, and cell without <a> tags
    if soup.table:
        rows = soup.table.findChildren(['tr'])
        for row in rows:
            cells = row.findChildren('td')
            out_line = []
            for cell in cells:
                out_line.append(unidecode(cell.text))
            out_str = '@'.join(out_line)
            fp.write(out_str + '\n')
    fp.close()
