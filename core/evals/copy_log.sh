CurrentDir=$(pwd)

for file in *;do
  if [[ "$file" == *"info"* ]];then
#    p=$(grep -o 'time_stamp=[^ ,]\+' $file)
#    folder=$(echo $p | sed "s/.*time_stamp='\(.*\)'.*/\1/g")
#
#    q=$(grep -o 'model=[^ ,]\+' $file)
#    model=$(echo $q | sed "s/.*model='\(.*\)'.*/\1/g")
#    if [[ ! " ${models[@]} " =~ " $model " ]]; then
#      models[${#models[@]}]=$model
#    fi
#
#    useful[${#useful[@]}]=$folder
  fi
done