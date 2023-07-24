import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict

plt.rcParams["font.family"] = "Times New Roman"


def rec_d():
    return defaultdict(rec_d)


def plot_line(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot"):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#b35806','#e08214','#fdb863','#8073ac','#542788']
    # linetype = ['-']
    linetype = ['-', '--']
    # markertype = ['o', '|', '+', 'x']
    #
    # X_max = float('inf')

    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        plt.plot(xs[i], data, linetype[i % len(linetype)],
                 color=colors[(i // 2) % len(colors)], label=linelabels[i],
                 linewidth=1.)
        # plt.plot(xs[i], data, linetype[i % len(linetype)],
        #          color=colors[i % len(colors)], label=linelabels[i],
        #          linewidth=1.)
        # X_max = min(X_max, max(xs[i]))

    # legend_properties = {'size': _fontsize}

    # plt.legend(prop=legend_properties,
        # frameon=False)
    if len(linelabels) > 1:
        plt.legend(ncol=2, frameon=False,
                   bbox_to_anchor=(1., -0.3), fontsize=7)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#    ax.tick_params(axis="y", direction="in")
#    ax.tick_params(axis="x", direction="in")

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    # plt.xlim(0, 20)
    # plt.ylim(0, 100)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(label, fontsize=_fontsize)
    plt.plot()
    plt.savefig(name, bbox_inches='tight')


def plot_line_2(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot"):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#b35806','#e08214','#fdb863','#8073ac','#542788']
    linetype = ['-']
    # markertype = ['o', '|', '+', 'x']
    #
    # X_max = float('inf')

    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        plt.plot(xs[i], data, linetype[i % len(linetype)],
                 color=colors[i % len(colors)], label=linelabels[i],
                 linewidth=1.)
        # X_max = min(X_max, max(xs[i]))

    legend_properties = {'size': _fontsize}

    plt.legend(prop=legend_properties,
        frameon=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#    ax.tick_params(axis="y", direction="in")
#    ax.tick_params(axis="x", direction="in")

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    # plt.xlim(0, 20)
    # plt.ylim(0, 100)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(label, fontsize=_fontsize)
    plt.plot()
    plt.savefig(name, bbox_inches='tight')


def run(task_prefix, task_dict, name_prefix):
    # personalized = "meta" # "meta" or "none" or "ditto"

    # x-axis
    training_round_list_list_no_sync = []
    testing_round_list_list_no_sync = []
    testing_hour_list_list = []

    # testing_acc_lines-axis
    acc_list_list = []
    acc_list_list_no_async = []
    duration_list_list_no_sync = []

    # label
    label_list = []
    label_list_no_async = []
    label_list_duration_no_async = []

    for k, v in task_dict.items():
        if v not in ["async", "local"]:
            label_list_duration_no_async.append(v)
            label_list_no_async.append(f"{v}_g")
            label_list_no_async.append(f"{v}_l")
        label_list.append(f"{v}_g")
        label_list.append(f"{v}_l")
        if v not in ["async", "local"]:
            label_list_duration_no_async.append(v)
            label_list_no_async.append(f"{v}")
        # label_list.append(f"{v}")

        testing_acc_lines = []
        testing_local_acc_lines = []
        training_duration_lines = []
        log_file = os.path.join(
            '..', '..', '..', 'core', 'evals', 'history', f"{task_prefix}_{k}",
            'aggregator', 'log_1'
        )
        with open(log_file, 'r') as file:
            lines = file.readlines()
        for idx, line in enumerate(lines):
            if 'FL Testing' in line:
                testing_acc_lines.append(line)
            elif 'FL Local Testing' in line:
                testing_local_acc_lines.append(line)
            elif 'Wall clock time' in line:
                training_duration_lines.append(line)

        # global testing accuracy
        round_list = []
        acc_list = []
        hour_list = []
        for f in testing_acc_lines:
            epoch = f[f.find("epoch")+7:].split(',')[0]
            top1_acc_str = f[f.find("top_1")+7:].split(' ')[0]
            virtual_clock_str = f[f.find("virtual_clock")+15:].split(',')[0]
            round_list.append(int(epoch))
            acc_list.append(float(top1_acc_str))
            hour_list.append(float(virtual_clock_str) / 60 / 60)

        acc_list_list.append(acc_list)
        if v not in ["async", "local"]:
            testing_round_list_list_no_sync.append(round_list)
            acc_list_list_no_async.append(acc_list)
        testing_hour_list_list.append(hour_list)

        # average local testing accuracy
        round_local_list = []
        acc_local_list = []
        hour_local_list = []
        for f in testing_local_acc_lines:
            epoch = f[f.find("epoch") + 7:].split(',')[0]
            top1_acc_str = f[f.find("top_1") + 7:].split(' ')[0]
            virtual_clock_str = f[f.find("virtual_clock") + 15:].split(',')[0]
            round_local_list.append(int(epoch))
            acc_local_list.append(float(top1_acc_str))
            hour_local_list.append(float(virtual_clock_str) / 60 / 60)

        acc_list_list.append(acc_local_list)
        if v not in ["async", "local"]:
            testing_round_list_list_no_sync.append(round_local_list)
            acc_list_list_no_async.append(acc_local_list)
        testing_hour_list_list.append(hour_local_list)

        # duration for each training round
        if v not in ["async", "local"]:
            round_list = []
            duration_list = []
            epoch = 0
            for f in training_duration_lines:
                t = int(f[f.find("Wall clock time") + 17:].split(',')[0])
                duration_list.append(t)
                if epoch > 0:
                    round_list.append(epoch)
                epoch += 1

            tmp_list = []
            for idx in range(len(duration_list[:-1])):
                tmp_list.append(duration_list[idx+1] - duration_list[idx])

            duration_list_list_no_sync.append(tmp_list)
            training_round_list_list_no_sync.append(round_list)


    # save_path = os.path.join(os.getcwd(), 'history', f"round_to_acc")
    # plot_line(acc_list_list_no_async, testing_round_list_list_no_sync,
    #           label_list_no_async, "Training Rounds", "Accuracy (%)", save_path)
    save_path = os.path.join(f"{name_prefix}_time_to_acc")
    plot_line(acc_list_list, testing_hour_list_list,
            label_list, " Training Time (h)", "Avg. Accuracy (%)", save_path)

    # save_path = os.path.join(os.getcwd(), 'history', f"round_duration")
    # plot_line_2(duration_list_list_no_sync, training_round_list_list_no_sync,
    #           label_list_duration_no_async, "Training Rounds", "Duration (s)", save_path)

task_dict = {
        f"random_none_all_test": "sync_plain",
    }
run("google_speech", task_dict, 'sync')

task_dict = {
        f"local_none_all_test_60": "local",
        f"random_none_all_test": "sync_plain",
    }
run("google_speech", task_dict, 'local')

task_dict = {
        f"random_none_all_test": "sync_plain",
        f"random_meta_all_test": f"sync_meta",
    }
run("google_speech", task_dict, 'meta')

task_dict = {
        f"random_none_all_test": "sync_plain",
        f"random_meta_all_test": f"sync_meta",
        f"async_meta_all_test_100_60": "async_meta",
    }
run("google_speech", task_dict, 'asyncmeta')

task_dict = {
        f"random_none_all_test": "sync_plain",
        f"async_none_all_test_60": "async_plain"
    }
run("google_speech", task_dict, 'async')

task_dict = {
        # f"local_none_all_test_60": "local",
        f"random_none_all_test": "sync_plain",
        # f"random_meta_all_test": f"sync_meta",
        # f"async_meta_all_test_100_60": "async_meta",
        f"async_none_all_test_60": "async_plain",
        f"random_none_larger_all_test": "sync_larger_plain",
    }
run("google_speech", task_dict, 'larger')