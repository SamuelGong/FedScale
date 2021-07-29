#!/bin/bash
original=$(pwd)
cd $1

for file in *;do
  if [[ "$file" == *"logging"* ]];then
    grep -v "Caffe2" $file > tmp && mv tmp $file
  fi
done

cd $original