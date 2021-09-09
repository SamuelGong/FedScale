import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

log_dir = os.path.join(os.getcwd(), 'history')
plt.rcParams["font.family"] = "Times New Roman"


def my_bar(x_label_list, y_list, x_label, y_label, save_path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#b35806', '#e08214', '#fdb863', '#8073ac', '#542788']

    l = len(y_list)
    bar_width = 0.8
    x_list = np.arange(l)
    text_list = ["{:.1f}".format(e) for e in y_list]
    # print(text_list)

    ax.bar(x_list, y_list, bar_width, color=colors[:l])
    for idx, y in enumerate(y_list):
        ax.text(x_list[idx] - bar_width / 3, y + 0.2, text_list[idx])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_list)
    ax.set_xticklabels(x_label_list, rotation=30)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, 1.1 * ymax)
    plt.savefig(save_path, bbox_inches='tight')


def run(task_prefix):
    task_dict = {
        f"local_none_all_test_60": "local",
        f"random_none_all_test": "sync_plain",
        f"random_meta_all_test": f"sync_meta",
        f"async_meta_all_test_100_60": "async_meta",
        f"async_none_all_test_60": "async_plain"
    }

    y_list = []
    x_label_list = []
    for k, v in task_dict.items():
        log_file = os.path.join(
            os.getcwd(), 'history', f"{task_prefix}_{k}",
            'aggregator', 'log_1'
        )
        with open(log_file, 'r') as file:
            lines = file.readlines()


        idx = lines[0].find("INFO")
        sts_str = lines[0][:idx - 1]
        sts = datetime.strptime(sts_str, "(%m-%d) %H:%M:%S")

        idx = lines[-1].find("INFO")
        ets_str = lines[-1][:idx - 1]
        ets = datetime.strptime(ets_str, "(%m-%d) %H:%M:%S")

        diff = ets - sts
        days = diff.total_seconds() / 60.0 / 60.0 / 24.0

        y_list.append(days)
        x_label_list.append(v)

    save_path = os.path.join(log_dir, 'exp_duration.png')
    y_label = "Duration (Days)"
    x_label = "Experiment"
    my_bar(x_label_list, y_list, x_label, y_label, save_path)

run(sys.argv[1])
