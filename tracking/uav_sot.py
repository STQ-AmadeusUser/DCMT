import _init_paths
import os
import pickle
import math
import numpy
import torch
import importlib
import argparse
import numpy as np
import torch.nn as nn
from easydict import EasyDict as edict

import torch.distributed as dist
import utils.read_file as reader
import utils.log_helper as recorder
import utils.model_helper as loader
import utils.lr_scheduler as learner
import utils.count_helper as counter
import utils.search_helper as searcher
import search.model_derived as model_derived
from models.densesiaminference import DenseSiamInference

from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataset.siamese_builder import SiameseDataset as data_builder
from core.trainer.siamese_train import siamese_train as trainer

import torch.backends.cudnn as cudnn


eps = 1e-5


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train DenseSiam')
    parser.add_argument('--cfg', type=str, default='../experiments/UAV.yaml', help='yaml configure file name')
    parser.add_argument('--arch_epoch', type=int, default=64, help='retrain densesiam from which epoch')
    parser.add_argument('--base_epoch', type=int, default=20, help='retrain uav from which epoch')
    args = parser.parse_args()

    return args


def main():
    # read config
    print('====> load configs <====')
    args = parse_args()
    config = edict(reader.load_yaml(args.cfg))
    os.environ['CUDA_VISIBLE_DEVICES'] = config.COMMON.GPUS
    if config.TRAIN.DDP.ISTRUE:
        dist.init_process_group(backend='nccl', init_method='env://')

    # create logger
    print('====> create logger <====')
    logger, _, tb_log_dir = recorder.create_logger(config, config.MODEL.NAME, 'retrain')
    # logger.info(pprint.pformat(config))
    logger.info(config)

    # create tensorboard logger
    print('====> create tensorboard <====')
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # create derived model of super model
    with open('./arch/arch_e{}.pkl'.format(str(args.arch_epoch)), 'rb') as f:
        arch_dict = pickle.load(f)
    arch_config = arch_dict['arch']
    derivedNetwork = getattr(model_derived, '%s_Net' % config.MODEL.NAME.upper())
    der_Net = lambda net_config: derivedNetwork(net_config, config=config)
    derived_model = der_Net(arch_config)
    logger.info("Derived Model Mult-Adds = %.2fMB" % counter.comp_multadds(derived_model))
    logger.info("Derived Model Num Params = %.2fMB", searcher.count_parameters_in_MB(derived_model))

    # create model
    print('====> build model <====')
    model = DenseSiamInference(derived_model, config)
    model = model.cuda()
    logger.info(model)

    # load pretrain
    # model = loader.load_pretrain(model, '../pretrain/{0}'.format(config.TRAIN.PRETRAIN), f2b=True)
    model = loader.load_retrain(model, './re_snapshot/checkpoint_e{}.pth'.format(args.base_epoch))

    # get optimizer
    if not config.TRAIN.START_EPOCH == config.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = learner.build_retrain_opt_lr(config, model, config.TRAIN.START_EPOCH)
    else:
        if config.TRAIN.RESUME:   # resume
            optimizer, lr_scheduler = learner.build_retrain_opt_lr(config, model, config.TRAIN.START_EPOCH)
        else:
            optimizer, lr_scheduler = learner.build_retrain_opt_lr(config, model, 0)  # resume wrong (last line)

    # check trainable again
    print('==========check trainable parameters==========')
    trainable_params = loader.check_trainable(model, logger)  # print trainable params info

    # resume or not
    if config.TRAIN.RESUME:   # resume
        model, optimizer, start_epoch, arch = loader.restore_from(model, optimizer, config.TRAIN.RESUME)
    else:
        start_epoch = config.TRAIN.START_EPOCH

    # parallel
    gpus = [int(i) for i in config.COMMON.GPUS.split(',')]
    gpu_num = world_size = len(gpus)  # or use world_size = torch.cuda.device_count()
    gpus = list(range(0, gpu_num))

    logger.info('GPU NUM: {:2d}'.format(len(gpus)))

    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')
    if not config.TRAIN.DDP.ISTRUE:
        model = DataParallel(model, device_ids=gpus).to(device)
    else:
        rank = config.TRAIN.DDP.RANK
        model = DistributedDataParallel(model, device_ids=[rank], output_device=rank)

    logger.info(lr_scheduler)
    logger.info('model prepare done')

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        # build dataloader, benefit to tracking
        train_set = data_builder(config)
        if not config.TRAIN.DDP.ISTRUE:
            train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH * gpu_num, num_workers=config.TRAIN.WORKERS,
                                      pin_memory=True, sampler=None, drop_last=True)
        else:
            sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True, seed=42)
            train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH * gpu_num, shuffle=False,
                                      num_workers=config.WORKERS, sampler=sampler, pin_memory=True, drop_last=True)

        # check if it's time to train backbone
        if epoch == config.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = learner.build_retrain_opt_lr(config, model.module, epoch)
            print('==========double check trainable==========')
            loader.check_trainable(model, logger)  # print trainable params info

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()

        inputs = {'data_loader': train_loader,
                  'model': model,
                  'optimizer': optimizer,
                  'device': device,
                  'epoch': epoch + 1,
                  'cur_lr': curLR,
                  'config': config,
                  'writer_dict': writer_dict,
                  'logger': logger
                  }

        model, writer_dict = trainer(inputs)

        # save model
        loader.save_model(model, epoch, optimizer, config.MODEL.NAME, config, isbest=False)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()




