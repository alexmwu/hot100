#!/bin/bash

# script to randomly select testing and training files
DECADES=(194 195 196 197 198 199 200 201)

for year in ${DECADES[@]}; do
  # random numbers to make testing files
  RN1=$((RANDOM % 10))
  RN2=$((RANDOM % 10))

  # ensure rn1 and rn2 are different
  while [ $RN1 -eq $RN2 ]; do
    RN2=$((RANDOM % 10))
  done

  for i in $(seq 1 10); do
    outfile=data/lyrics/$year$i
    outfile+=hot100.atsv
    if [ $i -eq $RN1 ] || [ $i -eq $RN2 ]; then
      cp $outfile data/sample_lyrics_test/
    else
      cp $outfile data/sample_lyrics_train/
    fi
  done

done

