import math
import time
import torch
import pdb
import utils.log_helper as recorder
import utils.model_helper as loader


def siamese_train(inputs):
    # parser inputs
    train_loader, model, optimizer, device = inputs['data_loader'], inputs['model'], inputs['optimizer'], inputs['device']
    epoch, cur_lr, cfg, writer_dict, logger = inputs['epoch'], inputs['cur_lr'], inputs['config'], inputs['writer_dict'], inputs['logger']

    # recorder
    batch_time = recorder.AverageMeter()
    data_time = recorder.AverageMeter()
    losses = recorder.AverageMeter()
    cls_losses = recorder.AverageMeter()
    reg_losses = recorder.AverageMeter()
    end = time.time()

    # switch to train mode
    model.train()
    model = model.to(device)

    for iter, batchinfo in enumerate(train_loader):
        data_time.update(time.time() - end)

        # SiamFC/SiamDW
        batch_keys = list(batchinfo.keys())
        template = batchinfo['template'].to(device)  # bx3x127x127
        search = batchinfo['search'].to(device)  # bx3x255x255
        cls_label = batchinfo['cls_label'].type(torch.FloatTensor).to(device)  # bx31x31

        # Ocean
        reg_label = batchinfo['reg_label'].float().to(device) if 'reg_label' in batch_keys else None  # bx31x31x4
        reg_weight = batchinfo['reg_weight'].float().to(device) if 'reg_weight' in batch_keys else None  # bx31x31

        # OceanPlus
        template_mask = batchinfo['template_mask'].to(device) if 'template_mask' in batch_keys else None  # bx127x127

        # AUtoMatch
        template_bbox = batchinfo['template_bbox'].to(device) if 'template_bbox' in batch_keys else None  # bx4
        search_bbox = batchinfo['search_bbox'].to(device) if 'search_bbox' in batch_keys else None  # bx4
        jitterBox = batchinfo['jitterBox'].float().to(device) if 'jitterBox' in batch_keys else None  # bx96x4
        jitter_ious = batchinfo['jitter_ious'].float().to(device) if 'jitter_ious' in batch_keys else None  #bx96

        model_inputs = {'template': template,
                        'search': search,
                        'cls_label': cls_label,
                        'reg_label': reg_label,
                        'reg_weight': reg_weight,
                        'template_bbox': template_bbox,
                        'search_bbox': search_bbox,
                        'template_mask': template_mask,
                        'jitterBox': jitterBox,
                        'jitter_ious': jitter_ious,
                        }

        model_loss = model(model_inputs)
        cls_loss = torch.mean(model_loss['cls_loss'])
        reg_loss = torch.mean(model_loss['reg_loss'])

        loss = cfg.TRAIN.CLS_WEIGHT * cls_loss + cfg.TRAIN.REG_WEIGHT * reg_loss
        loss = torch.mean(loss)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        if cfg.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if loader.is_valid_number(loss.item()):
            optimizer.step()

        # record loss
        loss = loss.item()
        losses.update(loss, template.size(0))

        cls_loss = cls_loss.item()
        cls_losses.update(cls_loss, template.size(0))

        reg_loss = reg_loss.item()
        reg_losses.update(reg_loss, template.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if (iter + 1) % cfg.TRAIN.PRINT_FREQ == 0:
            logger.info(
                '''Epoch: [{0}][{1}/{2}]\
                lr: {lr:.7f}\
                Batch Time: {batch_time.avg:.3f}s\
                Data Time:{data_time.avg:.3f}s
                REG_Loss:{reg_loss.avg:.5f}\
                CLS_Loss:{cls_loss.avg:.5f}\
                Loss:{loss.avg:.5f}'''.format(
                    epoch,
                    iter + 1,
                    len(train_loader),
                    lr=cur_lr,
                    batch_time=batch_time,
                    data_time=data_time,
                    reg_loss=reg_losses,
                    cls_loss=cls_losses,
                    loss=losses,
                    )
            )

            recorder.print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                        cfg.TRAIN.END_EPOCH * len(train_loader), logger)

        # write to tensorboard
        writer = writer_dict['writer']
        global_steps = writer_dict['train_global_steps']

        writer.add_scalar('loss', loss, global_steps)
        writer.add_scalar('cls_loss', cls_loss, global_steps)
        writer.add_scalar('reg_loss', reg_loss, global_steps)
        writer_dict['train_global_steps'] = global_steps + 1

    return model, writer_dict
