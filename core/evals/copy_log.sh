CurrentDir=$(pwd)
LogDir=~/models
DestDir=$CurrentDir/my_log
time_stamps=()
models=()
job_names=()

prune(){
  file=$1
  grep -v "Receive" $file > copy_tmp && mv copy_tmp $file
  grep -v "Start to train" $file > copy_tmp && mv copy_tmp $file
}

for file in *;do
  if [[ "$file" == *"info"* ]];then
    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
    time_stamp=$(echo $p | sed "s/.*time_stamp=\(.*\).*/\1/g")

    q=$(grep -o 'model=[^ ,]\+' $file)
    model=$(echo $q | sed "s/.*model=\(.*\).*/\1/g")

    r=$(grep -o 'job_name=[^ ,]\+' $file)
    job_name=$(echo $r | sed "s/.*job_name=\(.*\).*/\1/g")

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
  echo $dest

  cd $LogDir/${models[$i]}/${time_stamps[$i]}
  mkdir -p $dest/aggregator
  cd aggregator
  cp log $dest/aggregator
  prune $dest/aggregator/log

  mkdir -p $dest/executor
  cd ../executor

  for f in *;do
    if [[ "$f" == *"log"* ]];then
      cp $f $dest/executor
      prune $dest/executor/$f
    fi
  done
done

cd $CurrentDir