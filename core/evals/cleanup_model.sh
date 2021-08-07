#!/bin/bash

cd .
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    pair=$(sed -r "s/.*time_stamp=(\S+).*/\1/" $file)
    echo $pair
  fi
done