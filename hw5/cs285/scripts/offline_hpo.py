import argparse
import json

from cs285.scripts.run_hw5_offline import add_arguments, run_training_loop
from cs285.scripts.scripting_utils import make_logger, make_config, update_dict


def main():
    # copy argument parse from run_hw5_finetune.py.
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    parser.add_argument("--log_suffix", type=str, default='')
    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = f"offline_hpo_{args.log_suffix}"

    dataset = ['10k', '20k']
    cql_alpha = [0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]

    for ds in dataset:
        for alpha in cql_alpha:
            params = {'dataset_name': f"pointmass_hard_rnd{ds}", "cql_alpha": alpha}
            config = make_config(args.config_file, params)
            logger = make_logger(logdir_prefix, config, 'offline/cql')

            log_parameters = {}
            update_dict(log_parameters, config)
            logger.log_text('hyperparameters', json.dumps(log_parameters))

            run_training_loop(config, logger, args)


if __name__ == "__main__":
    main()