import _init_paths
import argparse
import importlib
import logging
import os
import sys
from easydict import EasyDict as edict
import torch
import torch.backends.cudnn as cudnn
import utils.read_file as reader


if __name__ == '__main__':

    parser = argparse.ArgumentParser("Params")
    parser.add_argument('--save', type=str, default='../experiments', help='experiment name')
    parser.add_argument('--meas_times', type=int, default=5000, help='measure times')
    parser.add_argument('--device', choices=['gpu', 'cpu'])
    parser.add_argument('--cfg', type=str, default='../experiments/DenseSiam.yaml', help='yaml configure tracking file')

    args = parser.parse_args()
    config = edict(reader.load_yaml(args.cfg))
    save_path = args.save

    input1_size = (config.TRAIN.BATCH, 3, config.TRAIN.TEMPLATE_SIZE, config.TRAIN.TEMPLATE_SIZE)
    input2_size = (config.TRAIN.BATCH, 3, config.TRAIN.SEARCH_SIZE, config.TRAIN.SEARCH_SIZE)
    input3_size = (config.TRAIN.BATCH, 4)
    input_size = (input1_size, input2_size, input3_size)

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(save_path, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    cudnn.benchmark = True
    cudnn.enabled = True

    SearchSpace = importlib.import_module('lib.search.search_space_' + config.MODEL.NAME.lower()).Network
    super_model = SearchSpace(config)
    super_model.eval()
    if args.device == 'gpu':
        super_model = super_model.cuda()

    latency_list, total_latency = super_model.get_cost_list(
        input_size, cost_type='latency',
        use_gpu=(args.device == 'gpu'),
        meas_times=args.meas_times
    )

    logging.info('latency_list:\n' + str(latency_list))
    logging.info('total latency: ' + str(total_latency) + 'ms')

    list_name = config.SEARCH.SUB_OBJ.LATENCY_LIST_PATH
    with open(os.path.join(save_path, list_name), 'w') as f:
        f.write(str(latency_list))
