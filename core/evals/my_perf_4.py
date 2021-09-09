import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict

plt.rcParams["font.family"] = "Times New Roman"
sstr = "clients to start at step"
estr = "clients to end at step"
stat_str = "clients online"
log_dir = os.path.join(os.getcwd(), 'history')


def plot_line(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot"):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#8073ac', '#b35806','#e08214','#fdb863']
    linetype = ['-']
    # linetype = ['-', '--']
    # markertype = ['o', '|', '+', 'x']
    #
    # X_max = float('inf')

    for i, data in enumerate(datas):
        plt.plot(xs[i], data, linetype[i % len(linetype)],
                 color=colors[i % len(colors)], label=linelabels[i],
                 linewidth=1.)
        # X_max = min(X_max, max(xs[i]))

    # legend_properties = {'size': _fontsize}

    # plt.legend(prop=legend_properties,
        # frameon=False)
    plt.legend(ncol=2, frameon=False,
               bbox_to_anchor=(1.1, -0.3), fontsize=7)

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
    ax.set_ylim(0)
    plt.plot()
    plt.savefig(name, bbox_inches='tight')


def run(task_prefix):
    task_dict = {
        f"local_none_all_test_60": "local",
        f"async_meta_all_test_100_60": "async_meta",
        f"async_none_all_test_60": "async_plain"
    }

    x_list_list = []
    y_list_list = []
    label_list = []
    x_max_task_name = None
    x_max = -1
    for task_name, v in task_dict.items():
        label = v
        log_file = os.path.join(
            os.getcwd(), 'history', f"{task_prefix}_{task_name}",
            'aggregator', 'log_1'
        )
        with open(log_file, 'r') as file:
            lines = file.readlines()

        useful_lines = []
        for l in lines:
            if sstr in l or estr in l:
                useful_lines.append(l)

        time_net_inc_dict = {}
        for l in useful_lines:
            idx = l.find("[Async]")
            l = l[idx + len("[Async] "):]
            idx = l.find("clients")
            num_clients = int(l[:idx - 1])

            idx = l.find("wall clock")
            time = round(float(l[idx + len("wall clock "):].split('s')[0]))

            if time not in time_net_inc_dict:
                time_net_inc_dict[time] = 0

            if sstr in l:  # increase
                time_net_inc_dict[time] += num_clients
            else:  # decrease
                time_net_inc_dict[time] -= num_clients

        y_list = []
        x_list = []
        for k in sorted(time_net_inc_dict.keys()):
            v = time_net_inc_dict[k]
            x_list.append(k / 60.0 / 60.0)  # secs to hours
            if len(y_list) == 0:
                y_list.append(v)
            else:
                y_list.append(y_list[-1] + v)

        y_max = max(y_list)
        if max(x_list) > x_max:
            x_max_task_name = task_name
            x_max = max(x_list)
        # y_min = min(y_list)
        # print(y_max, y_min)

        label += f' ({y_max})'
        label_list.append(label)
        x_list_list.append(x_list[0:-1])
        y_list_list.append(y_list[0:-1])

    # collect the actual number of active clients separately
    # print(x_max_task_name)
    log_file = os.path.join(
        os.getcwd(), 'history', f"{task_prefix}_{x_max_task_name}",
        'aggregator', 'log_1'
    )
    with open(log_file, 'r') as file:
        lines = file.readlines()

    useful_lines = []
    for l in lines:
        if stat_str in l:
            useful_lines.append(l)

    time_online_dict = {}
    for l in useful_lines:
        idx = l.find("Wall clock time:")
        l = l[idx + len("Wall clock time: "):]

        time = int(l.split(',')[0])
        l = l.split(',')[1]
        idx = l.find("clients online")
        num = int(l[:idx - 1])
        time_online_dict[time] = num

    x_list = []
    y_list = []
    for k in sorted(time_online_dict.keys()):
        x_list.append(k / 60.0 / 60.0)  # secs to hours
        y_list.append(time_online_dict[k])

    x_list_list.insert(0, x_list)
    y_list_list.insert(0, y_list)
    label_list.insert(0, 'available')

    # plot!
    save_path = os.path.join(log_dir, 'async_selected_clients.png')
    y_label = "# training clients"
    x_label = "Time (h)"
    plot_line(y_list_list, x_list_list, label_list, x_label, y_label, save_path)


run(sys.argv[1])