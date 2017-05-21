#!/usr/bin/env bash

input_file=$1
filelines=`cat $input_file`

for line in $filelines;
do
    echo $line
done