#!/bin/bash
# Inconsistent file names between frames & masks:
# frame name:	*raw.jpg
# mask name: 	*svg.png
# rename all frames ending with *raw.jpg --> *svg.png

path=$1 # directory to *_frames

# pattern for file name detection
pattern="*raw.jpg"

# pattern for substitution
mask_pattern="svg.png"

# rename frame files
for file in $path$pattern
do
	len1=${#file}
	len2=$((${#pattern}-1))
	mv $file  ${file::$(($len1-$len2))}$mask_pattern
done

rm $path".DS_Store"
