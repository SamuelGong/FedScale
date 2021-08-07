#!/bin/bash

CurrentDir=.
LogDir=~/models

cd .
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
    folder=$(echo $p | sed "s/.*time_stamp='\(.*\)'.*/\1/g")

    q=$(grep -o 'model=[^ ,]\+' $file)
    model=$(echo $q | sed "s/.*model='\(.*\)'.*/\1/g")


  fi
done