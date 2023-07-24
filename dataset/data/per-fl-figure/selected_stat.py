import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from collections import defaultdict
from datetime import datetime

plt.rcParams["font.family"] = "Times New Roman"
focus_str_in_agg_local = "FL Local Testing"
focus_str_in_exe_local = "(Local) After"
focus_str_in_agg_global = "FL Testing"
focus_str_in_exe_global = "] After"
log_dir = log_file = os.path.join('..', '..', '..', 'core', 'evals', 'history')


def rec_d():
    return defaultdict(rec_d)


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


def extract_executors_dict(folder, start_timestamp, end_timestamp, focus_str_in_exe):

    base_path = os.path.join(log_dir, folder, 'executor')
    file_paths = []
    for file_path in os.listdir(base_path):
        if 'log' in file_path:
            file_paths.append(file_path)

    result = {}
    for file_path in file_paths:
        log_path = os.path.join(base_path, file_path)

        with open(log_path, 'r') as f:
            ls = f.readlines()

        new_ls = []
        for idx, l in enumerate(ls):
            if focus_str_in_exe in l:
                new_ls.append(ls[idx - 1])
                new_ls.append(l)

        useful_ls = []
        for l in new_ls:
            idx = l.find('INFO')
            timestamp = l[:idx-1]
            if within_the_range(timestamp, start_timestamp, end_timestamp):
                useful_ls.append(l)

        for l in useful_ls:
            if 'test_accuracy' in l:
                idx = l.find("test_accuracy") + len("test_accuracy ")
                acc = float(l[idx:].split('%')[0])
                result[client_id] = acc
            else:
                idx = l.find("Rank") + len("Rank ")
                client_id = int(l[idx:].split(':')[0])

    return result


def plot_line(datas, xs, linelabels=None, label=None, y_label="CDF", name="my_plot", num_mod=1):
    _fontsize = 9
    fig = plt.figure(figsize=(2, 1.6), dpi=1200)  # 2.5 inch for 1/3 double column width
    ax = fig.add_subplot(111)

    colors = ['#b35806','#e08214','#fdb863','#8073ac','#542788']
    # linetype = ['-']
    linetype = ['-', '--', '-.', ':']
    # markertype = ['o', '|', '+', 'x']
    #
    # X_max = float('inf')

    X = [i for i in range(len(datas[0]))]
    for i, data in enumerate(datas):
        plt.plot(xs[i], data, linetype[i % num_mod],
                 color=colors[(i // num_mod) % len(colors)], label=linelabels[i],
                 linewidth=1.)
        # plt.plot(xs[i], data, linetype[i % len(linetype)],
        #          color=colors[i % len(colors)], label=linelabels[i],
        #          linewidth=1.)
        # X_max = min(X_max, max(xs[i]))

    # legend_properties = {'size': _fontsize}

    # plt.legend(prop=legend_properties,
        # frameon=False)
    # if len(linelabels) > 1:
    #     plt.legend(ncol=2, frameon=False,
    #                bbox_to_anchor=(1., -0.3), fontsize=7)
    plt.legend(frameon=False, fontsize=7)

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


def run(task_dict, name_prefix):

    x_list_list = []
    y_list_list = []
    label_list = []

    x_list_list_2 = []
    y_list_list_2 = []
    label_list_2 = []

    for k, v in task_dict.items():
        n_list = v['n_list']
        label = v['label']
        required_virtual_clock = v["virtual_clock"]
        folder = k
        log_file = os.path.join(
            '..', '..', '..', 'core', 'evals', 'history', k,
            'aggregator', 'log_1'
        )
        with open(log_file, 'r') as file:
            lines = file.readlines()

        useful_lines = []
        for idx, l in enumerate(lines):
            if "Selected" in l:
                useful_lines.append(lines[idx - 1])
                useful_lines.append(l)
            elif "clients to end" in l:  # async
                useful_lines.append(lines[idx - 3])
                useful_lines.append(l)

        client_time_dict = {}
        time_client_dict = {}
        max_part_num = -1
        for l in useful_lines:
            if "Selected" in l:
                idx = l.find("Selected participants to run: [") + len("Selected participants to run: [")
                l = l[idx:].split(']')[0]
                for s in l.split(','):
                    client_id = int(s)
                    if client_id not in client_time_dict:
                        client_time_dict[client_id] = [time]
                    else:
                        client_time_dict[client_id].append(time)

                    if time not in time_client_dict:
                        time_client_dict[time] = [client_id]
                    else:
                        time_client_dict[time].append(client_id)
            elif "clients to end" in l:
                idx = l.find("s: [") + len("s: [")
                l = l[idx:].split(']')[0]
                for s in l.split(','):
                    client_id = int(s)
                    if client_id not in client_time_dict:
                        client_time_dict[client_id] = [time]
                    else:
                        client_time_dict[client_id].append(time)

                    if time not in time_client_dict:
                        time_client_dict[time] = [client_id]
                    else:
                        time_client_dict[time].append(client_id)
            else:  # Wall clock time
                idx = l.find("Wall clock time:") + len("Wall clock time: ")
                time = int(l[idx:].split(',')[0])

        for k, v in client_time_dict.items():
            if max_part_num < len(v):
                max_part_num = len(v)
        print(f'Max # selected for a client: {max_part_num}')

        x_list = [i for i in sorted(time_client_dict.keys())]
        double_dict = rec_d()
        for t in x_list:
            for i in range(1, max_part_num+1):
                double_dict[t][i] = []
        for client_id, time_list in client_time_dict.items():
            for idx, time in enumerate(time_list):
                if not idx == len(time_list) - 1:
                    for t in x_list:
                        if time <= t < time_list[idx + 1]:
                            double_dict[t][idx + 1].append(client_id)
                else:
                    for t in x_list:
                        if time <= t:
                            double_dict[t][idx + 1].append(client_id)

        for n in n_list:
            y_list = []
            for t in x_list:
                tmp = 0
                for a in range(n, max_part_num + 1):
                    tmp += len(double_dict[t][a])
                y_list.append(tmp)
            new_y_list = []
            new_x_list = []
            for idx, y in enumerate(y_list):
                if y == 0:
                    continue
                else:
                    new_x_list.append(x_list[idx] / 60.0 / 60.0) # secs to hours
                    new_y_list.append(y)
            label_list.append(label + r' ($\geq$' + f'{n})')
            x_list_list.append(new_x_list)
            y_list_list.append(new_y_list)

        ######

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

        temp = sorted(virtual_clock_timestamp_dict.keys())
        if required_virtual_clock < 0:
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

        client_local_acc_dict = extract_executors_dict(folder, start_timestamp, end_timestamp, focus_str_in_exe_local)
        last_time = x_list[-1]
        x_list_2 = []
        y_list_2 = []
        for num in double_dict[last_time].keys():
            cnt = 0
            avg = 0.0
            for k, v in double_dict[last_time].items():
                if k < num:
                    continue
                cnt += len(v)
                avg += sum([client_local_acc_dict[i] for i in v])

            if cnt == 0:
                continue
            avg /= cnt

            x_list_2.append(num)
            y_list_2.append(avg)

        x_list_list_2.append(x_list_2)
        y_list_list_2.append(y_list_2)
        label_list_2.append(label)

    x_label = "Timeline (h)"
    y_label = "# Clients"
    save_path = os.path.join(f'{name_prefix}_selected_clients')
    plot_line(y_list_list, x_list_list, linelabels=label_list,
              label=x_label, y_label=y_label, name=save_path, num_mod=len(n_list))

    x_label = "Minimal # Participation"
    y_label = "Avg. Local Accuracy (%)"
    save_path = os.path.join(f'{name_prefix}_part_acc')
    plot_line(y_list_list_2, x_list_list_2, linelabels=label_list_2,
              label=x_label, y_label=y_label, name=save_path)


# task = "google_speech"
# task_dict = {
#         # f"{task}_local_none_all_test_60" : {
#         #     "virtual_clock": -1,
#         #     "label": "local"
#         # },
#         # f"{task}_random_none_all_test": {
#         #     "virtual_clock": -1,
#         #     "label": "sync_plain"
#         # },
#         f"{task}_random_meta_all_test": {
#             "virtual_clock": -1,
#             "label": "sync_meta"
#         },
#         # f"{task}_async_meta_all_test_100_60": {
#         #     "virtual_clock": -1,
#         #     "label": "async_meta"
#         # },
#         # f"{task}_async_none_all_test_60": {
#         #     "virtual_clock": -1,
#         #     "label": "async_plain"
#         # },
#     }
# run(task_dict, 'meta')

# task = "google_speech"
# task_dict = {
#         # f"{task}_local_none_all_test_60" : {
#         #     "virtual_clock": -1,
#         #     "label": "local"
#         # },
#         # f"{task}_random_none_all_test": {
#         #     "virtual_clock": -1,
#         #     "label": "sync_plain"
#         # },
#         f"{task}_random_meta_all_test": {
#             "virtual_clock": -1,
#             "label": "sync_meta",
#             "n_list": [1, 60, 120]
#         },
#         f"{task}_async_meta_all_test_100_60": {
#             "virtual_clock": -1,
#             "label": "async_meta",
#             "n_list": [1, 100, 1500]
#         },
#         # f"{task}_async_none_all_test_60": {
#         #     "virtual_clock": -1,
#         #     "label": "async_plain"
#         # },
#     }
# run(task_dict, 'asyncmeta')

# task = "google_speech"
# task_dict = {
#         # f"{task}_local_none_all_test_60" : {
#         #     "virtual_clock": -1,
#         #     "label": "local"
#         # },
#         f"{task}_random_none_all_test": {
#             "virtual_clock": -1,
#             "label": "sync_plain",
#             "n_list": [100]
#         },
#         # f"{task}_random_meta_all_test": {
#         #     "virtual_clock": -1,
#         #     "label": "sync_meta",
#         #     "n_list": [1, 60, 120]
#         # },
#         # f"{task}_async_meta_all_test_100_60": {
#         #     "virtual_clock": -1,
#         #     "label": "async_meta",
#         #     "n_list": [1, 100, 1500]
#         # },
#         f"{task}_async_none_all_test_60": {
#             "virtual_clock": -1,
#             "label": "async_plain",
#             "n_list": [100]
#         },
#     }
# run(task_dict, 'async')

task = "google_speech"
task_dict = {
        # f"{task}_local_none_all_test_60" : {
        #     "virtual_clock": -1,
        #     "label": "local"
        # },
        f"{task}_random_none_all_test": {
            "virtual_clock": -1,
            "label": "sync_plain",
            "n_list": [100]
        },
        # f"{task}_random_meta_all_test": {
        #     "virtual_clock": -1,
        #     "label": "sync_meta",
        #     "n_list": [1, 60, 120]
        # },
        # f"{task}_async_meta_all_test_100_60": {
        #     "virtual_clock": -1,
        #     "label": "async_meta",
        #     "n_list": [1, 100, 1500]
        # },
        f"{task}_async_none_all_test_60": {
            "virtual_clock": -1,
            "label": "async_plain",
            "n_list": [100]
        },
        f"{task}_random_none_larger_all_test": {
            "virtual_clock": -1,
            "label": "sync_larger_plain",
            "n_list": [100]
        },
    }
run(task_dict, 'larger')