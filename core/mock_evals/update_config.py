import sys
from ruamel.yaml import YAML

yaml = YAML()
yaml.explicit_start = True
yaml.indent(mapping=4)

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin)
    return data


def dump_yaml_conf(yaml_file, obj):
    with open(yaml_file, 'w') as fout:
        yaml.dump(obj, fout)


def run(args):
    file, key, value, value_type = args
    data = load_yaml_conf(file)
    found = False
    if value_type == "int":
        pair = {key: int(value)}
    else:
        pair = {key: value}
    for idx, item in enumerate(data['job_conf']):
        if key in item:
            data['job_conf'][idx] = pair
            found = True
            break
    if not found:
        data['job_conf'].append(pair)

    dump_yaml_conf(file, data)

run(sys.argv[1:])