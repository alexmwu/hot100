#!/bin/sh

# ls does this (and is better with an awk cmd that selects size columns of 0)
# but this is faster
for f in data/*; do
  if ! [ -s $f ]; then
    echo $f
  fi
done