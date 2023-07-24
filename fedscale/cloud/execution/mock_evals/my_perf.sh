#!/bin/sh
tasks=("openimage" "reddit" "google_speech")

for task in "${tasks[@]}"; do
#    python my_perf.py ${task} comm_ratio
    python my_perf.py ${task} what_if_comm_speed_up
    python my_perf.py ${task} what_if_comp_speed_up
#    python my_perf.py ${task} comm_ratio_pdf
done
