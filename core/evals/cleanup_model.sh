#!/bin/bash

CurrentDir=$(pwd)
LogDir=~/models
useful=()

cd $CurrentDir
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
    folder=$(echo $p | sed "s/.*time_stamp='\(.*\)'.*/\1/g")

    q=$(grep -o 'model=[^ ,]\+' $file)
    model=$(echo $q | sed "s/.*model='\(.*\)'.*/\1/g")

    useful[${#useful[@]}]=$folder
  fi
done

for folder in ${useful[@]}; do
  echo $folder
done

cd $LogDir/$model

for file in *;do
  if [[ "$folder" == *"logging"* ]];then
    if [[ ! " ${useful[@]} " =~ " $folder " ]]; then
      echo $folder
    fi
  fi
done

cd $CurrentDir