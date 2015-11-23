#!/usr/bin/env python

# Script to get accuracy on test set results
# Function in bagOfWords not currently working

import sys

NUM, ARIST, SONG, YEAR, DECADE = range(5)

if len(sys.argv) < 2:
    print "Test output csv file as arg"
    sys.exit(1)
filename = sys.argv[1]
nSamples = 0.0
nIncorrect = 0.0
with open(filename, 'r') as f:
	next(f)
	for line in f:
		attr = line.strip('\n').split('@')
		if attr[DECADE] == '':
			nIncorrect += 1
		elif int(float(attr[YEAR])//10*10) != int(float(attr[DECADE])):
			#print line
			nIncorrect += 1
		nSamples += 1
print nIncorrect, nSamples
print 1-(nIncorrect/nSamples)
