#!/usr/bin/env python

# Scrapes the site http://billboardtop100of.com/ for all Billboard
# Hot 100 year end chart songs (e.g., all Hot 100 songs in the year
# end chart for 2014)


import urllib2
from datetime import date
from bs4 import BeautifulSoup

fp = open("hot100.csv","w")

bbSite = 'http://billboardtop100of.com/'

startYear = 1940

currYear = date.today().year
# iterate to the current year - 1
for y in xrange(startYear, currYear):
    pass
y = startYear
# for whatever reason, the link to 1940 is 336
if y == 1940:
    y = 336
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
# check if the td has an <a> tag
            links = cell.findChildren('a')
            if len(links) == 0:
                out_line.append(cell.text)
            else:
                for link in links:
                    out_line.append(link.text)
        out_str = ','.join(out_line).encode('ascii', 'ignore')
        fp.write(out_str + '\n')
# no table (only seen in 1940)

fp.close()

    # for tr in soup.table:
# # handle cells with <a> tags
        # links = soup.findChildren('a')
        # if len(links) == 0:
            # print tr
        # else:
            # for link in links:
                # print link
