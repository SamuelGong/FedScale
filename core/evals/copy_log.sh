CurrentDir=$(pwd)

LogDir=~/models
DestDir=CurrentDir/log
time_stamps=()
models=()
job_names=()

for file in *;do
  if [[ "$file" == *"info"* ]];then
    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
    time_stamp=$(echo $p | sed "s/.*time_stamp=\(.*\).*/\1/g")

    q=$(grep -o 'model=[^ ,]\+' $file)
    model=$(echo $q | sed "s/.*model=\(.*\).*/\1/g")

    r=$(grep -o 'job_name=[^ ,]\+' $file)
    job_name=$(echo $q | sed "s/.*job_name=\(.*\).*/\1/g")

    time_stamps[${#time_stamps[@]}]=$time_stamp
    models[${#models[@]}]=$model
    job_names[${#job_names[@]}]=$job_name
  fi
done

arraylength=${#job_names[@]}
for (( i=0; i<${arraylength}; i++ ));
do
  job_name="${job_names[$i]}"
  dest=${DestDir}/${job_name}

  cd $LogDir/${models[$i]}/${time_stamps[$i]}
  mkdir -p $dest/aggregator
  cd aggregator
  cp log $dest/aggregator

  mkdir -p $dest/executor
  cd ../executor

  for file in *;do
    if [[ "$file" == *"log"* ]];then
      cp $file $dest/executor
    fi
  done
done

cd $CurrentDir