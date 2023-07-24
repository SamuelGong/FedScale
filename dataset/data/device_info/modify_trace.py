import sys
import os
import pickle
import numpy as np


def run(file):
    postfix = "_homo"
    new_file = file + postfix
    with open(file, 'rb') as f:
        d = pickle.load(f)

    if file == "client_behave_trace":
        max_finish_time = 0
        for _, v in d.items():
            finish_time = v['finish_time']
            if finish_time > max_finish_time:
                max_finish_time = finish_time

        for k in d.keys():
            d[k]['duration'] = max_finish_time
            d[k]['finish_time'] = max_finish_time
            d[k]['active'] = [0]
            d[k]['inactive'] = [max_finish_time]

    else:
        comp = []
        comm = []
        for _, v in d.items():
            comp.append(v["computation"])
            comm.append(v["communication"])

        comp_median = np.median(comp)
        comm_median = np.median(comm)

        for k in d.keys():
            d[k]["computation"] = comp_median
            d[k]["communication"] = comm_median

    with open(new_file, "wb") as f:
        pickle.dump(d, f)

run(sys.argv[1])