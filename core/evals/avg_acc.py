import sys
import ast

def run(task):
    log_file = f'{task}_logging'
    with open(log_file, 'r') as file:
        lines = file.readlines()

    result_dict = dict()
    iter_idx = 1
    for idx, line in enumerate(lines):
        if 'Selected participants' in line:
            idx = line.find("run:")
            l = ast.literal_eval(line[idx+5:].split(":")[0])
            for client_id in l:
                if client_id in result_dict:
                    result_dict[client_id].append(iter_idx)
                else:
                    result_dict[client_id] = [iter_idx]
            iter_idx += 1

    # for k in sorted(result_dict.keys()):
    #     print(f"{k}: {result_dict[k]}")

run(sys.argv[1])