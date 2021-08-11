#!/bin/bash

if [ -d $1 ]; then
  cd $1
  original=$(pwd)

  for file in *;do
    if [[ "$file" == *"logging"* ]];then
      grep -v "Caffe2" $file > suppress_tmp && mv suppress_tmp $file
    fi
  done

  cd $original
elif [ -f $1 ]; then
  grep -v "Caffe2" $1 > suppress_tmp && mv suppress_tmp $1
fi