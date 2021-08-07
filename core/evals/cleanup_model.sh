#!/bin/bash

cd .
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    A=$(grep -o 'time_stamp=[^ ,]\+' $file)
    echo $A
  fi
done