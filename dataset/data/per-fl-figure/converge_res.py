import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

log_dir = os.path.join('..', '..', '..', 'core', 'evals', 'history')
focus_str_in_agg_local = "FL Local Testing"
focus_str_in_exe_local = "(Local) After"
focus_str_in_agg_global = "FL Testing"
focus_str_in_exe_global = "] After"
plt.rcParams["font.family"] = "Times New Roman"


def within_the_range(ts_str, sts_str, ets_str):
    ts = datetime.strptime(ts_str, "(%m-%d) %H:%M:%S")
    sts = None
    if sts_str:
        sts = datetime.strptime(sts_str, "(%m-%d) %H:%M:%S")
    ets = datetime.strptime(ets_str, "(%m-%d) %H:%M:%S")

    if sts:
        if sts <= ts <= ets:
            return True
        else:
            return False
    else:
        if ts <= ets:
            return True
        else:
            return False


def extract_executors(folder, start_timestamp, end_timestamp, focus_str_in_exe):

    base_path = os.path.join(log_dir, folder, 'executor')
    file_paths = []
    for file_path in os.listdir(base_path):
        if 'log' in file_path:
            file_paths.append(file_path)

    result = []
    for file_path in file_paths:
        log_path = os.path.join(base_path, file_path)

        with open(log_path, 'r') as f:
            ls = f.readlines()

        new_ls = []
        for l in ls:
            if focus_str_in_exe in l:
                new_ls.append(l)

        useful_ls = []
        for l in new_ls:
            idx = l.find('INFO')
            timestamp = l[:idx-1]
            if within_the_range(timestamp, start_timestamp, end_timestamp):
                useful_ls.append(l)

        for l in useful_ls:
            idx = l.find("test_accuracy") + len("test_accuracy ")
            result.append(float(l[idx:].split('%')[0]))

    return result


def my_cdf(datas, labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#b35806','#e08214','#fdb863','#8073ac','#542788']
    linetype = ['-', '-.', '--', ':']

    for idx, data in enumerate(datas):
        data_sorted = sorted(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        ax.plot(data_sorted, p, linetype[idx%len(linetype)], label=labels[idx], color=colors[idx%len(colors)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 100)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.legend(loc='best', fontsize=9, frameon=False)
    plt.savefig(path, bbox_inches='tight')


def run(task_dict, name_prefix):
    # personalized = "meta"

    test_accs_local_list = []
    test_accs_global_list = []
    label_list = []
    for folder, d in task_dict.items():
        agg_path = os.path.join(log_dir, folder, 'aggregator', 'log_1')
        with open(agg_path, 'r') as f:
            lines = f.readlines()

        focus_lines = []
        for l in lines:
            if focus_str_in_agg_local in l:
                focus_lines.append(l)

        virtual_clock_timestamp_dict = {}
        for f in focus_lines:
            idx = f.find('virtual_clock')
            virtual_clock_str = int(f[idx + len('virtual_clock: '):].split('.')[0])
            idx = f.find('INFO')
            time_stamp_str = f[:idx - 1]
            virtual_clock_timestamp_dict[virtual_clock_str] = time_stamp_str

        required_virtual_clock = d["virtual_clock"]
        temp = sorted(virtual_clock_timestamp_dict.keys())
        if d["virtual_clock"] < 0:
            required_virtual_clock = temp[-1]
        end_timestamp = virtual_clock_timestamp_dict[required_virtual_clock]

        end_timestamp_idx = None
        for idx, virtual_clock in enumerate(temp):
            if virtual_clock == required_virtual_clock:
                end_timestamp_idx = idx
                break
        assert end_timestamp_idx

        start_timestamp_idx = end_timestamp_idx - 1
        if start_timestamp_idx < 0:
            start_timestamp = None
        else:
            start_timestamp = virtual_clock_timestamp_dict[temp[start_timestamp_idx]]

        test_accs_local = extract_executors(folder, start_timestamp, end_timestamp, focus_str_in_exe_local)
        test_accs_local_list.append(test_accs_local)

        test_accs_global = extract_executors(folder, start_timestamp, end_timestamp, focus_str_in_exe_global)
        test_accs_global_list.append(test_accs_global)

        label_list.append(d["label"])

    new_test_accs_global_list = []
    new_label_list = []
    for i, l in enumerate(label_list):
        if 'plain' not in l:
            new_label_list.append(l)
            new_test_accs_global_list.append(test_accs_global_list[i])
    if len(new_label_list) > 0:
        save_path_global = os.path.join(f'{name_prefix}_global_converge_acc_cdf.png')
        y_label = "CDF across clients"
        x_label = "Global Testing Acc. (%)"
        my_cdf(new_test_accs_global_list, new_label_list, x_label, y_label, save_path_global)

    save_path_local = os.path.join(f'{name_prefix}_local_converge_acc_cdf.png')
    y_label = "CDF across clients"
    x_label = "Local Testing Acc. (%)"
    my_cdf(test_accs_local_list, label_list, x_label, y_label, save_path_local)


task = "google_speech"
task_dict = {
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        },
    }
run(task_dict, 'sync')

task = "google_speech"
task_dict = {
        f"{task}_local_none_all_test_60" : {
            "virtual_clock": -1,
            "label": "local"
        },
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        }
    }
run(task_dict, 'local')

task = "google_speech"
task_dict = {
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        },
        f"{task}_random_meta_all_test": {
            "virtual_clock": -1,
            "label": "sync_meta"
        },
    }
run(task_dict, 'meta')

task = "google_speech"
task_dict = {
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        },
        f"{task}_random_meta_all_test": {
            "virtual_clock": -1,
            "label": "sync_meta"
        },
        f"{task}_async_meta_all_test_100_60": {
            "virtual_clock": -1,
            "label": "async_meta"
        },
    }
run(task_dict, 'asyncmeta')

task = "google_speech"
task_dict = {
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        },
        f"{task}_async_none_all_test_60": {
            "virtual_clock": -1,
            "label": "async_plain"
        },
    }
run(task_dict, 'async')

task = "google_speech"
task_dict = {
        # f"{task}_local_none_all_test_60" : {
        #     "virtual_clock": -1,
        #     "label": "local"
        # },
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain"
        },
        # f"{task}_random_meta_all_test": {
        #     "virtual_clock": -1,
        #     "label": "sync_meta"
        # },
        # f"{task}_async_meta_all_test_100_60": {
        #     "virtual_clock": -1,
        #     "label": "async_meta"
        # },
        f"{task}_async_none_all_test_60": {
            "virtual_clock": -1,
            "label": "async_plain"
        },
        f"{task}_random_none_larger_all_test": {
            "virtual_clock": -1,
            "label": "sync_larger_plain"
        },
    }
run(task_dict, 'larger')