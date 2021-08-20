#!/bin/bash

CurrentDir=$(pwd)
LogDir=~/models

job_names=()
models=()
time_stamps=()

cd $CurrentDir
for file in *;do
  if [[ "$file" == *"info"* ]];then
    p=$(grep -o 'job_name=[^ ]\+' $file)
    job_name=$(echo $p | sed -e 's#.*=\(\)#\1#')

    q=$(grep -o 'model=[^ ]\+' $file)
    model=$(echo $q | sed -e 's#.*=\(\)#\1#')

    r=$(grep -o 'time_stamp=[^ ]\+' $file)
    time_stamp=$(echo $r | sed -e 's#.*=\(\)#\1#')

    job_names[${#job_names[@]}]=$job_name
    if [[ ! " ${models[@]} " =~ " $model " ]]; then
      models[${#models[@]}]=$model
    fi
    time_stamps[${#time_stamps[@]}]=$time_stamp
  fi
done

# Step 1: remove folders in ~/models
for model in "${models[@]}";do
  cd $LogDir/$model

  for folder in *;do
    if [[ ! " ${time_stamp[@]} " =~ " $folder " ]]; then
#      rm -rf $folder
      echo $folder
    fi
  done
done

cd $CurrentDir

# Step 2: remove temporary log files at current folder
exit  # only uncomment it when needed
for file in *;do
  if [[ "$file" == *"logging"* ]];then
    rm $file
done