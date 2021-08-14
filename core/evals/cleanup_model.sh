#!/bin/bash
exit
CurrentDir=$(pwd)
LogDir=~/models
useful=()
models=()

cd $CurrentDir
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
    folder=$(echo $p | sed "s/.*time_stamp='\(.*\)'.*/\1/g")

    q=$(grep -o 'model=[^ ,]\+' $file)
    model=$(echo $q | sed "s/.*model='\(.*\)'.*/\1/g")
    if [[ ! " ${models[@]} " =~ " $model " ]]; then
      models[${#models[@]}]=$model
    fi

    useful[${#useful[@]}]=$folder
  fi
done


for model in "${models[@]}";do
  cd $LogDir/$model

  for folder in *;do
    if [[ ! " ${useful[@]} " =~ " $folder " ]]; then
      rm -rf $folder
    fi
  done
done

cd $CurrentDir