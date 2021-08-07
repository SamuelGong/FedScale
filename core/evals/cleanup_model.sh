#!/bin/bash

cd .
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    pair=$(grep -o 'time_stamp=[^ ,]\+' $file)
    folder=$(echo $pair | sed "s/.*time_stamp='\(.*\)'.*/\1/g")
    echo $folder
  fi
done