#!/bin/bash

for f in data/charts/*.atsv; do python getLyricsParallel.py "$f" & done
