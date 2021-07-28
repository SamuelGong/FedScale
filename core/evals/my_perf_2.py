import matplotlib.pyplot as plt
import numpy as np
import ast
from collections import defaultdict
import sys


def rec_d():
    return defaultdict(rec_d)


def my_pdf(datas, labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#b35806','#f1a340', '#998ec3','#542788']
    linetype = ['-', '-.', '--', ':']
    num_bins = 30

    for idx, data in enumerate(datas):
        weights = np.ones_like(data) / len(data)
        ax.hist(data, bins=num_bins, weights=weights, histtype='step',
                label=labels[idx], color=colors[idx%len(colors)])


    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best', fontsize=6)
    plt.savefig(path, bbox_inches='tight')


def my_cdf(datas, labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#b35806','#f1a340', '#998ec3','#542788']
    linetype = ['-', '-.', '--', ':']

    for idx, data in enumerate(datas):
        data_sorted = sorted(data)
        p = 1. * np.arange(len(data)) / (len(data) - 1)
        ax.plot(data_sorted, p, linetype[idx%len(linetype)], label=labels[idx], color=colors[idx%len(colors)])

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best', fontsize=6)
    plt.savefig(path, bbox_inches='tight')


def my_stacked(x, datas, xticklabels,
               legend_labels, xlabel, ylabel, path):
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)
    ax = fig.add_subplot(111)

    colors = ['#f1a340','#fee0b6','#d8daeb','#998ec3','#542788']

    processed_datas = []
    for idx, data in enumerate(datas):
        if idx == 0:
            processed_datas.append(data)
        else:
            processed_datas.append(data - datas[idx - 1])

    current_bottom = np.zeros(len(xticklabels))
    for idx, row in enumerate(processed_datas):

        ax.bar(x, height=row, bottom=current_bottom,
               color=colors[idx], label=legend_labels[idx])
        current_bottom += row
        if idx == len(processed_datas) - 1:
            for xi, bi in zip(x, current_bottom):
                ax.text(xi - 0.3, bi * 1.01, "{:.2f}".format(bi))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(1, ax.get_ylim()[1])
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.legend(loc='best', fontsize=6)
    plt.savefig(path, bbox_inches='tight')

def my_parse(log_file):
    with open(log_file, 'r') as file:
        lines = file.readlines()

    focus = []
    for idx, line in enumerate(lines):
        if 'Selected participants to run' in line:
            focus.append(lines[idx + 1])

    epoch_clients_perf = []
    for f in focus:
        f = f.split('\n')[0]
        p = ast.literal_eval(f)
        epoch_clients_perf.append(p)
    return epoch_clients_perf

# plot_task = "comm_ratio"
# plot_task = "what_if_comm_speed_up"
# plot_task = "what_if_comp_speed_up"
round_limit = 499
def run(task, plot_task):
    # task = "openimage"
    # task = "reddit"
    # task = "google_speech"
    plt.rcParams["font.family"] = "Times New Roman"
    num_clients = 100
    local_steps_list = np.arange(2, 10, 2)
    t_dict_dict = rec_d()

    for local_steps in local_steps_list:
        log_file = f'{task}_logging_ls{local_steps}'
        epoch_clients_perf = my_parse(log_file)

        r_cnt = 0
        for r in range(len(epoch_clients_perf)):
            if round_limit is not None and r_cnt > round_limit:
                break

            t_dict_dict[local_steps][r]['comp'] = []
            t_dict_dict[local_steps][r]['comm'] = []
            t_dict_dict[local_steps][r]['client_id'] = []
            t_dict_dict[local_steps][r]['total'] = []
            for k, v in epoch_clients_perf[r].items():
                tcomp = v["computation"]
                tcomm = v["communication"]
                t = tcomp + tcomm
                t_dict_dict[local_steps][r]['comp'].append(tcomp)
                t_dict_dict[local_steps][r]['comm'].append(tcomm)
                t_dict_dict[local_steps][r]['total'].append(t)
                t_dict_dict[local_steps][r]['client_id'].append(k)
            r_cnt += 1

    if plot_task == "comm_ratio" or plot_task == "comm_ratio_pdf":
        data_list = []
        for local_steps in local_steps_list:
            comm_portion_list = []
            for r in sorted(t_dict_dict[local_steps].keys()):
                if len(t_dict_dict[local_steps][r]['total']) >= num_clients:
                    top_k = num_clients - 1
                else:
                    top_k = -1
                sorted_idx = np.argsort(t_dict_dict[local_steps][r]['total'])[top_k]
                t_dict_dict[local_steps][r]['sorted_idx'] = sorted_idx

                topk_duration = round(t_dict_dict[local_steps][r]['total'][sorted_idx])
                topk_comp = round(t_dict_dict[local_steps][r]['comp'][sorted_idx])
                topk_comm = round(t_dict_dict[local_steps][r]['comm'][sorted_idx])
                comm_portion = topk_comm / topk_duration
                comm_portion_list.append(comm_portion)
            data_list.append(comm_portion_list)

        label_list = [str(i) for i in local_steps_list]
        if plot_task == "comm_ratio":
            my_cdf(data_list, label_list, 'Comm. time portion', 'CDF across rounds', f'{task}_comm_portion.png')
        else:
            my_pdf(data_list, label_list, 'Comm. time portion', 'Portion across rounds', f'{task}_comm_portion_pdf.png')

    elif plot_task == "what_if_comm_speed_up" or plot_task == "what_if_comp_speed_up":
        speed_up_factor_list = [1] + list(np.arange(20, 120, 20))

        raw_data_list = []
        for f in speed_up_factor_list:
            bars = []
            for local_steps in local_steps_list:
                total_time = 0
                for r in sorted(t_dict_dict[local_steps].keys()):
                    if len(t_dict_dict[local_steps][r]['total']) >= num_clients:
                        top_k = num_clients - 1
                    else:
                        top_k = -1
                    sorted_idx = np.argsort(t_dict_dict[local_steps][r]['total'])[top_k]

                    tcomp = t_dict_dict[local_steps][r]['comp']
                    tcomm = t_dict_dict[local_steps][r]['comm']

                    if plot_task == "what_if_comm_speed_up":
                        f1 = 1.0
                        f2 = f
                    else:
                        f1 = f
                        f2 = 1.0

                    tcomp = np.array(tcomp) / f1
                    tcomm = np.array(tcomm) / f2
                    what_if_t = tcomp + tcomm
                    what_if_sorted_idx = np.argsort(what_if_t)[top_k]

                    topk_duration = round(what_if_t[what_if_sorted_idx])
                    total_time += topk_duration
                bars.append(total_time)
            bars = np.array(bars)
            raw_data_list.append(bars)

        data_list = []
        for idx, row in enumerate(raw_data_list):
            if idx == 0:
                continue
            data_list.append(raw_data_list[0] / row)

        xtick_labels = [str(i) for i in local_steps_list]
        legend_labels = [str(i) for i in speed_up_factor_list[1:]]

        if plot_task == "what_if_comm_speed_up":
            my_stacked(np.arange(len(data_list[0])), data_list,
                       xtick_labels, legend_labels, "# local steps",
                       "Comm. speedup factor", f'{task}_what_if_speedup_comm.png')
        else:
            my_stacked(np.arange(len(data_list[0])), data_list,
                       xtick_labels, legend_labels, "# local steps",
                       "Comp. speedup factor", f'{task}_what_if_speedup_comp.png')


run(sys.argv[1], sys.argv[2])