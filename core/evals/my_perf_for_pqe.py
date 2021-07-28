import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

plt.rcParams["font.family"] = "Times New Roman"


def rec_d():
    return defaultdict(rec_d)


def my_hist(datas, labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#998ec3','#542788']
    num_bins = 30

    mean_list = []
    for idx, data in enumerate(datas):
        mean = np.mean(data)
        mean_list.append(mean)
        ax.axvline(x=mean, ymin=0, color=colors[1])
        ax.hist(data, bins=num_bins, label=labels[idx], color=colors[idx%len(colors)])

    xmi, xma = ax.get_xlim()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    r = 1.2
    ymi, yma = ax.get_ylim()
    ax.set_ylim(ymi, yma * r)

    for idx, data in enumerate(datas):
        mean = mean_list[idx]
        ax.text(0.5*(mean + xmi)*0.72, yma, 'mean: {:.2f} min'.format(mean))

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # ax.legend(loc='best', fontsize=6)
    plt.savefig(path, bbox_inches='tight')


def plot_line(datas, xs, linelabels=None, label=None,
              y_label="CDF", name="my_plot", mode=0, _type=-1):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#b35806','#f1a340', '#998ec3','#542788']
    linetype = ['-', '--', '-.', '-', '-', ':']
    markertype = ['o', '|', '+', 'x']

    X_max = float('inf')

    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        _type = max(_type, i)
        ax.plot(xs[i], data, linetype[_type % len(linetype)], color=colors[mode], label=linelabels[i],
                 linewidth=1.)
        X_max = min(X_max, max(xs[i]))

    legend_properties = {'size': _fontsize}

    # plt.legend(prop=legend_properties, frameon=False)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#    ax.tick_params(axis="y", direction="in")
#    ax.tick_params(axis="x", direction="in")

    plt.yticks(fontsize=_fontsize)
    plt.xticks(fontsize=_fontsize)

    plt.xlim(0)
    r = 1.2
    ymi, yma = ax.get_ylim()
    ax.set_ylim(ymi, yma * r)

    xmi, xma = ax.get_xlim()
    ax.set_xlim(xmi, xma * r)

    for i, data in enumerate(datas):
        ax.axhline(y=data[-1], xmin=0, xmax=xs[i][-1]/xma/r, color='#e0e0e0', dashes=[1, 1, 1, 1])
        ax.axvline(x=xs[i][-1], ymin=0, ymax=data[-1]/yma/r, color='#e0e0e0', dashes=[1, 1, 1, 1])
        ax.text(xs[i][-1]/2 * 0.1, data[-1] * 1.1, "target accuracy: {:.2f}%".format(data[-1]))
        if mode == 0:
            ax.text(xs[i][-1] * 1.05, data[-1] / 2 * 0.5, "time: {:.2f}h".format(xs[i][-1]), rotation=270)
        elif mode == 1:
            ax.text(xs[i][-1] * 1.05, data[-1] / 2 * 0.5, "round: {}".format(xs[i][-1]), rotation=270)

    # plt.ylim(10)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(label, fontsize=_fontsize)
    plt.plot()
    plt.savefig(name, bbox_inches='tight')

def run(task):
    num_clients = 100
    local_step_list = np.arange(2, 4, 2)
    label_list = [str(i) for i in local_step_list]

    x_list_list = []
    y_list_list = []
    all_list = []
    culmu_time_in_hrs_list = []
    for local_step in local_step_list:
        log_file = f'{task}_logging_ls{local_step}'
        with open(log_file, 'r') as file:
            lines = file.readlines()

        all = []
        base = 0
        focus = []
        test_flag = 0
        culmu_time_in_hrs = []
        for idx, line in enumerate(lines):
            if 'Wall clock:' in line:
                culmu_time = line[line.find("Wall clock:")+12:].split(' ')[0]
                temp = int(culmu_time)
                all.append((temp - base)/60)
                base = temp

                if test_flag == 1:
                    test_flag = 0
                    culmu_time_in_hrs.append(int(culmu_time) / 60 / 60)
            if 'FL Testing' in line:
                focus.append(line)
                test_flag = 1
        culmu_time_in_hrs_list.append(culmu_time_in_hrs)
        all_list.append(all)

        x_list = []
        y_list = []
        for f in focus:
            epoch = f[f.find("epoch")+7:].split(',')[0]
            top1_acc_str = f[f.find("top_1")+7:].split(' ')[0]
    #        print(epoch, top1_acc_str)
            x_list.append(int(epoch))
            y_list.append(float(top1_acc_str))

        y_list_list.append(y_list)
        x_list_list.append(x_list)

    plot_line(y_list_list, x_list_list,
              label_list, "Training Rounds", "Accuracy (%)", "pqe_time_to_round", 1)
    plot_line(y_list_list, culmu_time_in_hrs_list,
            label_list, " Training Time (h)", "Accuracy (%)", f"pqe_time_to_acc", 0)

    for idx, all in enumerate(all_list):
        round = x_list_list[idx][-1]
        all_list[idx] = all[1:round+1]
    my_hist(all_list, label_list, 'Round Latency (min)', '# Rounds', f'pqe_round_latency')


run(sys.argv[1])