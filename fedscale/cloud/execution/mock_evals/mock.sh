#!/bin/bash

conf_file=configs/reddit/conf.yml

for (( local_steps=20; local_steps<=100; local_steps=local_steps+20 ))
do
  python update_config.py ${conf_file} local_steps ${local_steps} int
  python manager.py submit ${conf_file}

  pids=($(ps -ef | grep python | grep FedScale | awk '{ print $2 }'))
  sec=0
  for pid in ${pids[*]}; do
    while ps -p ${pid} >/dev/null 2>&1; do
       echo "waiting for local_steps: ${local_steps} to complete... ${sec} s"
       sleep 10
       ((sec=sec+10))
    done
  done
  mv reddit_logging reddit_logging_ls${local_steps}
done