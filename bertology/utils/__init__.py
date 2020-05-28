
from argparse import ArgumentParser, Namespace

def dict_to_args(d, parser: ArgumentParser):
    argv = []
    for flag, value in d.items():
        if value is True:
            argv.append(f'--{flag}')
        else:
            argv.extend((f'--{flag}', str(value)))
    args, argv = parser.parse_known_args(argv)
    return args

def args_to_dict(args: Namespace):
    return args.__dict__
