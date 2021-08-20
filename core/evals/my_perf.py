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

    colors = ['#b35806', '#f1a340', '#998ec3', '#542788']
    linetype = ['-', '--']
    # markertype = ['o', '|', '+', 'x']
    #
    # X_max = float('inf')

    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        plt.plot(xs[i], data, linetype[i % len(linetype)],
                 color=colors[(i // 2) % len(colors)], label=linelabels[i],
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

def plot_line_2(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot"):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#b35806', '#f1a340', '#998ec3', '#542788']
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

def run(task_prefix):
    personalized = "none" # "meta" or "none" or "ditto"
    task_dict = {
        f"random_{personalized}_all_test": "random",
        f"oort_{personalized}_all_test": "oort"
    }
    label_list = []
    duration_label_list = []

    x_list_list = []
    y_list_list = []
    x_list_list_2 = []
    y_list_list_2 = []
    culmu_time_in_hrs_list = []

    for k, v in task_dict.items():
        if v not in ["async"]:
            duration_label_list.append(v)
        label_list.append(f"{v}_g")
        label_list.append(f"{v}_l")
        log_file = os.path.join(
            os.getcwd(), 'history', f"{task_prefix}_{k}",
            'aggregator', 'log_1'
        )
        with open(log_file, 'r') as file:
            lines = file.readlines()

        focus = []
        focus_local = []
        focus_duration = []
        test_flag = 0
        culmu_time_in_hrs = []
        for idx, line in enumerate(lines):
            if 'Wall clock:' in line:
                culmu_time = line[line.find("Wall clock:")+12:].split(' ')[0]
                if test_flag == 1:
                    test_flag = 0
                    culmu_time_in_hrs.append(int(culmu_time) / 60 / 60)
            if 'FL Testing' in line:
                focus.append(line)
                test_flag = 1
            elif 'FL Local Testing' in line:
                focus_local.append(line)
            elif 'Wall clock time' in line:
                focus_duration.append(line)
        # culmu_time_in_hrs = culmu_time_in_hrs[:10]
        culmu_time_in_hrs_list.append(culmu_time_in_hrs)
        culmu_time_in_hrs_list.append(culmu_time_in_hrs)

        x_list = []
        y_list = []
        # for f in focus[:10]:
        for f in focus:
            epoch = f[f.find("epoch")+7:].split(',')[0]
            top1_acc_str = f[f.find("top_1")+7:].split(' ')[0]
            x_list.append(int(epoch))
            y_list.append(float(top1_acc_str))
        y_list_list.append(y_list)
        x_list_list.append(x_list)

        x_local_list = []
        y_local_list = []
        # for f in focus_local[:10]:
        for f in focus_local:
            epoch = f[f.find("epoch") + 7:].split(',')[0]
            top1_acc_str = f[f.find("top_1") + 7:].split(' ')[0]
            x_local_list.append(int(epoch))
            y_local_list.append(float(top1_acc_str))
        y_list_list.append(y_local_list)
        x_list_list.append(x_local_list)

        if v not in ["async"]:
            x_list_2 = []
            y_list_2 = []
            epoch = 0
            for f in focus_duration:
                t = int(f[f.find("Wall clock time") + 17:].split(',')[0])
                y_list_2.append(t)
                if epoch > 0:
                    x_list_2.append(epoch)
                epoch += 1

            tmp_list = []
            for idx in range(len(y_list_2[:-1])):
                tmp_list.append(y_list_2[idx+1] - y_list_2[idx])

            y_list_list_2.append(tmp_list)
            x_list_list_2.append(x_list_2)


    save_path = os.path.join(os.getcwd(), 'history', f"{personalized}_round_to_acc")
    plot_line(y_list_list, x_list_list,
              label_list, "Training Rounds", "Accuracy (%)", save_path)
    save_path = os.path.join(os.getcwd(), 'history', f"{personalized}_time_to_acc")
    plot_line(y_list_list, culmu_time_in_hrs_list,
            label_list, " Training Time (h)", "Accuracy (%)", save_path)

    save_path = os.path.join(os.getcwd(), 'history', f"{personalized}_round_duration")
    plot_line_2(y_list_list_2, x_list_list_2,
              duration_label_list, "Training Rounds", "Duration (s)", save_path)

run(sys.argv[1])
