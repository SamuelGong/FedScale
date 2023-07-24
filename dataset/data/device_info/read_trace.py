import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams["font.family"] = "Times New Roman"


def load_client_profile(file_path):
    # load client profiles
    global_client_profile = {}
    if os.path.exists(file_path):
        with open(file_path, 'rb') as fin:
            # {clientId: [computer, bandwidth]}
            global_client_profile = pickle.load(fin)
    return global_client_profile


def my_cdf(datas, labels, xlabel, ylabel, path, xscale=None):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#000000']
    # colors = ['#b35806', '#f1a340', '#998ec3', '#542788']
    linetype = ['-', '-.', '--', ':']

    for idx, data in enumerate(datas):
        data_sorted = sorted(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        ax.plot(data_sorted, p, linewidth=1, color=colors[idx%len(colors)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xscale is not None:
        ax.set_xscale(xscale)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.savefig(path, bbox_inches='tight')


def my_density(datas, labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    # colors = ['#b35806','#f1a340', '#998ec3','#542788']
    linetype = ['-', '-.', '--', ':']
    num_bins = 30

    for idx, data in enumerate(datas):
        # weights = np.ones_like(data) / len(data)
        # ax.hist(data, bins=num_bins, histtype='stepfilled', density=True, color=colors[idx % len(colors)])
        sns.distplot(data, hist=True, kde=True,
                     bins=num_bins, color='#000000', hist_kws={'color': '#cccccc'},
                     kde_kws={'linewidth': 1})

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if labels is not None:
        ax.legend(loc='best', fontsize=6)
    plt.savefig(path, bbox_inches='tight')


def plot_line(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot", _type=-1):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['black', 'orange', 'blueviolet', 'slateblue', 'DeepPink',
              '#FF7F24', 'blue', 'blue', 'blue', 'red', 'blue', 'red', 'red', 'grey', 'pink']
    linetype = ['-', '--', '-.', '-', '-', ':']
    markertype = ['o', '|', '+', 'x']

    X_max = float('inf')

    X = [i for i in range(len(datas[0]))]

    for i, data in enumerate(datas):
        _type = max(_type, i)
        plt.plot(xs[i], data, linetype[_type % len(linetype)], color=colors[i % len(colors)], label=linelabels[i],
                 linewidth=1.)
        X_max = min(X_max, max(xs[i]))

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
    plt.ylim(0)

    ax.set_ylabel(y_label, fontsize=_fontsize)
    ax.set_xlabel(label, fontsize=_fontsize)
    plt.savefig(name, bbox_inches='tight')


def run(file_path):
    trace = load_client_profile(file_path)
    print(list(trace.keys())[0], trace[list(trace.keys())[0]])
    # print(list(trace.keys())[1], trace[list(trace.keys())[1]])
    # print(list(trace.keys())[2], trace[list(trace.keys())[2]])
    # print(list(trace.keys())[-1], trace[list(trace.keys())[-1]])

    # if file_2_path
    exit(0)

    if 'behave' in file_path:
        # device_models = set()
        # for k, v in trace.items():
        #     device_models.add(v['model'])

        #print(device_models)
        # print(f"device num: {len(list(trace.keys()))}, device model num: {len(device_models)}")
        hours = 96
        secs = hours * 60 * 60
        stats = np.zeros(secs+1, dtype=int)
        online = np.zeros(secs+1, dtype=int)
        duration = []
        for k, v in trace.items():
            active = v['active']
            inactive = v['inactive']
            d = 0
            for a, i in zip(active, inactive):
                stats[a:i+1] += 1
                if i <= secs:
                    d += (i - a) / 60 / 60
                elif a < secs:
                    d += (secs - a) / 60 / 60
            duration.append(d)
            finish = v['finish_time']
            online[0:finish+1] += 1

        percent = stats / online * 100
        my_density([duration], None, "Duration of Avail. (h)", "Density", 'intra-device')
        plot_line([stats], [np.arange(secs+1)/60/60], ["Avalilable clients"],
                  "Timeline (h)", "# Clients", "inter-device")
        plot_line([percent], [np.arange(secs+1)/60/60], ["Avalilable clients"],
                  "Timeline (h)", "Percent of Clients (%)", "inter-device-portion")
    else:
        latencies = []
        throughputs = []
        for _, v in trace.items():
            latencies.append(float(v['computation']))
            throughputs.append(float(v['communication']))

        my_cdf([latencies], None, 'Compute Latency (ms)', 'CDF across Clients', 'compute-latency', 'log')
        my_cdf([throughputs], None, 'Network Throughput (kbps)', 'CDF across Clients', 'network-throughput', 'log')
        print(f"# entries in capacity model: {len(list(trace.keys()))}")

run(sys.argv[1])

