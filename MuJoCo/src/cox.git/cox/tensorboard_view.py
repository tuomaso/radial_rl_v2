from .readers import CollectionReader
from argparse import ArgumentParser
import os
import re

def has_tensorboard(dirname):
    if not os.path.isdir(dirname):
        return False
    return 'tensorboard' in os.listdir(dirname)

def main():
    parser = ArgumentParser(description='Helper script for starting interpretable tensorboard sessions.')
    parser.add_argument('--logdir', type=str, required=True, help="logdir (Same as tensorboard)")
    parser.add_argument('--params', nargs="+", required=True, help="Parameters to show in the job name")
    parser.add_argument('--port', type=int, required=False, default=6006, help="Port (passed on to tensorboard")
    parser.add_argument('--metadata-table', type=str, default="metadata", help="Name of the metadata table")
    parser.add_argument('--format-str', type=str, required=True,  \
                    help="How to format the job name prefix (the suffix is always the uid)")
    parser.add_argument('--script-location', type=str, default='.')
    parser.add_argument('--filter-param', action='append', nargs=2, help='Format: {parameter} {required value regex}')
    args = parser.parse_args()

    reader = CollectionReader(args.logdir)
    metadata_df = reader.df(args.metadata_table)

    # Find all of the valid experiments, i.e. folders with a tensorboard/ directory inside of them
    subdirs = filter(lambda x: has_tensorboard(os.path.join(args.logdir, x)), os.listdir(args.logdir))

    parameter_filters = {}
    if args.filter_param is not None:
        parameter_filters = dict(args.filter_param)

    tensorboard_arg_str = ""
    for experiment_id in subdirs:
        params_to_fill = []
        try:
            for p in args.params:
                param_value = metadata_df[metadata_df['exp_id'] == experiment_id][p][0]
                if p in parameter_filters and re.match(parameter_filters[p], param_value) is None:
                    raise ValueError("Filtered out---this exception will be caught.")
                if type(param_value) == list:
                    param_value = '.'.join([str(x) for x in param_value])

                params_to_fill.append(param_value)
            name_str = args.format_str.format(*params_to_fill) + "---" + experiment_id
            tensorboard_arg_str += "%s:%s," % (name_str, os.path.join(args.logdir, experiment_id))
        except IndexError as ie:
            print("Warning: Skipping experiment %s" % (experiment_id,))
        except ValueError as ve:
            pass


    full_job_str = "tensorboard --logdir %s --port %d" % (tensorboard_arg_str, args.port)
    open(os.path.join(args.script_location, 'tensorboard-view.sh'), 'w').write(full_job_str)

if __name__ == "__main__":
    main()
