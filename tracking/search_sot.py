import _init_paths
import os
import torch
import argparse
import importlib
from easydict import EasyDict as edict

import torch.distributed as dist
import utils.read_file as reader
import utils.log_helper as recorder
import utils.model_helper as loader
import utils.lr_scheduler as learner
import utils.count_helper as counter
import utils.search_helper as searcher
import search.model_derived as model_derived

from tensorboardX import SummaryWriter
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from dataset.siamese_builder import SiameseDataset as data_builder
from core.trainer.densesiam_train import DenseSiamTrainer


eps = 1e-5


def parse_args():
    """
    args for training.
    """
    parser = argparse.ArgumentParser(description='Train SiamCorrs')
    parser.add_argument('--cfg', type=str, default='../experiments/DenseSiam.yaml', help='yaml configure file name')
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
    logger, _, tb_log_dir = recorder.create_logger(config, config.MODEL.NAME, 'search')
    # logger.info(pprint.pformat(config))
    logger.info(config)

    # create tensorboard logger
    print('====> create tensorboard <====')
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
    }

    # create model
    print('====> build super model <====')
    SearchSpace = importlib.import_module('lib.search.search_space_' + config.MODEL.NAME.lower()).Network
    ArchGenerater = importlib.import_module('lib.search' + '.derive_arch_' + config.MODEL.NAME.lower()).ArchGenerate
    derivedNetwork = getattr(model_derived, '%s_Net' % config.MODEL.NAME.upper())
    super_model = SearchSpace(config)
    arch_gener = ArchGenerater(super_model, config)
    der_Net = lambda net_config: derivedNetwork(net_config, config=config)
    super_model = DataParallel(super_model)

    super_model = super_model.cuda()
    logger.info(super_model)
    betas, head_alphas, stack_alphas = super_model.module.display_arch_params()
    derived_archs = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
    derived_model = der_Net('|'.join(map(str, derived_archs)))
    logger.info("Derived Model Mult-Adds = %.2fMB" % counter.comp_multadds(derived_model))
    logger.info("Derived Model Num Params = %.2fMB", searcher.count_parameters_in_MB(derived_model))

    # load pretrain
    super_model = loader.load_pretrain(super_model, '../pretrain/{0}'.format(config.TRAIN.PRETRAIN), f2b=True, adm=True)

    # load sub obj
    if config.SEARCH.SUB_OBJ.TYPE == 'latency':
        with open(os.path.join('../experiments', config.SEARCH.SUB_OBJ.LATENCY_LIST_PATH), 'r') as f:
            latency_list = eval(f.readline())
        super_model.module.sub_obj_list = latency_list
        logger.info("Super Network latency (ms) list: \n")
        logger.info(str(latency_list))
    else:
        raise NotImplementedError
    logger.info("Num Params = %.2fMB", searcher.count_parameters_in_MB(super_model))

    # get optimizer
    if not config.TRAIN.START_EPOCH == config.TRAIN.UNFIX_EPOCH:
        optimizer, lr_scheduler = learner.build_densesiam_opt_lr(config, super_model, config.TRAIN.START_EPOCH)
    else:
        if config.TRAIN.RESUME:   # resume
            optimizer, lr_scheduler = learner.build_densesiam_opt_lr(config, super_model, config.TRAIN.START_EPOCH)
        else:
            optimizer, lr_scheduler = learner.build_densesiam_opt_lr(config, super_model, 0)  # resume wrong (last line)
    weight_optimizer = optimizer
    arch_optimizer = torch.optim.Adam(
        [{'params': super_model.module.arch_alpha_params, 'lr': config.SEARCH.ARCH.ALPHA_LR},
         {'params': super_model.module.arch_beta_params, 'lr': config.SEARCH.ARCH.BETA_LR}],
        betas=(0.5, 0.999),
        weight_decay=config.SEARCH.ARCH.WEIGHT_DECAY)

    # check trainable again
    print('==========check trainable parameters==========')
    trainable_params = loader.check_trainable(super_model, logger)  # print trainable params info

    # resume or not
    if config.TRAIN.RESUME:   # resume
        super_model, optimizer, start_epoch, arch = loader.restore_from_search(super_model, optimizer, config.TRAIN.RESUME)
    else:
        start_epoch = config.TRAIN.START_EPOCH

    # parallel
    gpus = [int(i) for i in config.COMMON.GPUS.split(',')]
    gpu_num = world_size = len(gpus)  # or use world_size = torch.cuda.device_count()
    gpus = list(range(0, gpu_num))
    logger.info('GPU NUM: {:2d}'.format(len(gpus)))
    device = torch.device('cuda:{}'.format(gpus[0]) if torch.cuda.is_available() else 'cpu')

    logger.info(lr_scheduler)
    logger.info('super model prepare done')

    # start training
    trainer = DenseSiamTrainer(config, gpus, device)
    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):

        # check if it's time to train backbone
        if epoch == config.TRAIN.UNFIX_EPOCH:
            logger.info('training backbone')
            optimizer, lr_scheduler = learner.build_densesiam_opt_lr(config, super_model, epoch)
            print('==========double check trainable==========')
            loader.check_trainable(super_model, logger)  # print trainable params info
            weight_optimizer = optimizer

        lr_scheduler.step(epoch)
        curLR = lr_scheduler.get_cur_lr()

        # build dataloader, benefit to tracking
        if epoch < config.SEARCH.ARCH_EPOCH:
            train_set = data_builder(config)
            train_loader = DataLoader(train_set, batch_size=config.TRAIN.BATCH * gpu_num,
                                      num_workers=config.TRAIN.WORKERS,
                                      pin_memory=True, sampler=None, drop_last=True)
            inputs = {'data_loader': train_loader,
                      'model': super_model,
                      'weight_optimizer': weight_optimizer,
                      'arch_optimizer': arch_optimizer,
                      'epoch': epoch + 1,
                      'cur_lr': curLR,
                      'writer_dict': writer_dict,
                      'logger': logger
                      }
        else:
            tracking_dataset = data_builder(config)
            train_size = int(0.9 * len(tracking_dataset))
            val_size = len(tracking_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(tracking_dataset, [train_size, val_size])
            train_loader = DataLoader(train_dataset, batch_size=config.TRAIN.BATCH * gpu_num,
                                      num_workers=config.TRAIN.WORKERS,
                                      pin_memory=True, sampler=None, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=config.TRAIN.BATCH * gpu_num,
                                    num_workers=config.TRAIN.WORKERS,
                                    pin_memory=True, sampler=None, drop_last=True)
            data_loaders = (train_loader, val_loader)
            inputs = {'data_loader': data_loaders,
                      'model': super_model,
                      'weight_optimizer': weight_optimizer,
                      'arch_optimizer': arch_optimizer,
                      'epoch': epoch + 1,
                      'cur_lr': curLR,
                      'writer_dict': writer_dict,
                      'logger': logger
                      }

        super_model, writer_dict = trainer(inputs)

        betas, head_alphas, stack_alphas = super_model.module.display_arch_params()
        derived_arch = arch_gener.derive_archs(betas, head_alphas, stack_alphas)
        derived_arch_str = '|\n'.join(map(str, derived_arch))
        derived_model = der_Net(derived_arch_str)
        derived_flops = counter.comp_multadds(derived_model)
        derived_params = searcher.count_parameters_in_MB(derived_model)
        logger.info("Derived Model Mult-Adds = %.2fMB" % derived_flops)
        logger.info("Derived Model Num Params = %.2fMB" % derived_params)
        writer_dict['writer'].add_scalar('derived_flops', derived_flops, epoch)
        epoch_derived_model = {'epoch': epoch,
                               'multadds': derived_flops,
                               'params': derived_params,
                               'arch': derived_arch_str}

        # save model
        loader.save_model(super_model, epoch, optimizer, config.MODEL.NAME, config, isbest=False)
        loader.save_arch(epoch_derived_model, config)

    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
