import matplotlib.pyplot as plt
import numpy as np
import sys
from collections import defaultdict

plt.rcParams["font.family"] = "Times New Roman"


def rec_d():
    return defaultdict(rec_d)


def plot_line(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot", _type=-1):
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
        plt.plot(xs[i], data, linetype[_type % len(linetype)], color=colors[i % len(colors)], label=linelabels[i],
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

    plt.xlim(0)
    # plt.ylim(10)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(label, fontsize=_fontsize)
    plt.plot()
    plt.savefig(name, bbox_inches='tight')

def run(task):
    total_workers = 200
    local_step_list = np.arange(2, 10, 2)
    label_list = [str(i) for i in local_step_list]

    x_list_list = []
    y_list_list = []
    culmu_time_in_hrs_list = []
    for local_step in local_step_list:
        log_file = f'{task}_logging_ls{local_step}_tw{total_workers}'
        with open(log_file, 'r') as file:
            lines = file.readlines()

        focus = []
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
        culmu_time_in_hrs_list.append(culmu_time_in_hrs)

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
              label_list, "Training Rounds", "Accuracy (%)", f"{task}_round_to_acc_tw{total_workers}")
    plot_line(y_list_list, culmu_time_in_hrs_list,
            label_list, " Training Time (h)", "Accuracy (%)", f"{task}_time_to_acc_tw{total_workers}")


run(sys.argv[1])