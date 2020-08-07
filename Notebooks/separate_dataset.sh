#!/bin/bash
# separate images staining nuclei & staining membrane

root_path=$1 # root directory of the dataset
subdir=$2
orig_path=$1$2
nuclei_path=$1$2"_nuclei"
membrane_path=$1$2"_membrane"

# create new subdirectory for membrane images
mkdir $membrane_path

# pattern for separation
membrane_files=(`find $orig_path -name "*f.png" -o -name "*F_2UL.png"`)

for file in "${membrane_files[@]}"
do
	fname=$(echo $file | awk -F/ '{print $NF}')
	mv $orig_path"/"$fname $membrane_path"/"$fname
done

# rename original subdirectory to "nuclei"
mv $orig_path $nuclei_path

# cleanup
#rm $nuclei_path"/.DS_Store"
#rm $membrane_path"/.DS_Store"
