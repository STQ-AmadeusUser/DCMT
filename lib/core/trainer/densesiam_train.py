import math
import time
import torch
import importlib
import utils.log_helper as recorder
import utils.model_helper as loader
from torch.nn import DataParallel
from search.dropped_model import Dropped_Network


class DenseSiamTrainer(object):
    def __init__(self, config, gpus, device):
        self.weight_sample_num = config.SEARCH.WEIGHT_SAMPLE_NUM
        loss_module = importlib.import_module('lib.models.loss')
        cls_loss_type = config.MODEL.LOSS.CLS_LOSS
        reg_loss_type = config.MODEL.LOSS.REG_LOSS
        self.cls_criterion = getattr(loss_module, cls_loss_type)
        self.reg_criterion = getattr(loss_module, reg_loss_type)
        self.Dropped_Network = lambda model: Dropped_Network(model, softmax_temp=config.SEARCH.SOFTMAX_TEMP)
        self.config = config
        self.gpus = gpus
        self.device = device

    def __call__(self, inputs):
        # parser inputs
        model, epoch, cur_lr = inputs['model'], inputs['epoch'], inputs['cur_lr']
        writer_dict, logger = inputs['writer_dict'], inputs['logger']
        self.weight_optimizer = inputs['weight_optimizer']
        self.arch_optimizer = inputs['arch_optimizer']
        if isinstance(inputs['data_loader'], tuple):
            train_loader, val_loader = inputs['data_loader'][0], inputs['data_loader'][1]
            search_stage = 1
        else:
            train_loader, val_loader = inputs['data_loader'], None
            search_stage = 0

        # recorder
        batch_time = recorder.AverageMeter()
        data_time = recorder.AverageMeter()
        ious = recorder.AverageMeter()
        losses = recorder.AverageMeter()
        cls_losses = recorder.AverageMeter()
        reg_losses = recorder.AverageMeter()
        sub_objs = recorder.AverageMeter()
        end = time.time()

        # switch to train mode
        model.train()
        model = model.to(self.device)

        if search_stage == 1:
            self.set_param_grad_state('Arch')
            assert val_loader is not None
            for iter, batchinfo in enumerate(val_loader):
                data_time.update(time.time() - end)

                template = batchinfo['template'].to(self.device)  # bx3x127x127
                search = batchinfo['search'].to(self.device)  # bx3x255x255
                cls_label = batchinfo['cls_label'].type(torch.FloatTensor).to(self.device)  # bx31x31
                reg_label = batchinfo['reg_label'].float().to(self.device)  # bx31x31x4
                reg_weight = batchinfo['reg_weight'].float().to(self.device)  # bx31x31
                template_bbox = batchinfo['template_bbox'].to(self.device)  # bx4

                model_inputs = {'template': template,
                                'search': search,
                                'cls_label': cls_label,
                                'reg_label': reg_label,
                                'reg_weight': reg_weight,
                                'template_bbox': template_bbox,
                                }
                iou, loss, cls_loss, reg_loss, sub_obj = self.arch_step(model_inputs, model, search_stage)

                # record loss
                ious.update(iou, template.size(0))
                losses.update(loss, template.size(0))
                cls_losses.update(cls_loss, template.size(0))
                reg_losses.update(reg_loss, template.size(0))
                sub_objs.update(sub_obj, template.size(0))
                batch_time.update(time.time() - end)
                end = time.time()

                if (iter + 1) % self.config.TRAIN.PRINT_FREQ == 0:
                    logger.info(
                        '''Epoch: [{0}][{1}/{2}]\
                        lr: {lr:.7f}\
                        Batch Time: {batch_time.avg:.3f}s\
                        Data Time:{data_time.avg:.3f}s
                        REG_Loss:{reg_loss.avg:.5f}\
                        CLS_Loss:{cls_loss.avg:.5f}\
                        Loss:{loss.avg:.5f}\
                        IOU:{iou.avg:.5f}\
                        Sub_Obj:{sub_obj.avg:.5f}'''.format(
                            epoch,
                            iter + 1,
                            len(val_loader),
                            lr=cur_lr,
                            batch_time=batch_time,
                            data_time=data_time,
                            reg_loss=reg_losses,
                            cls_loss=cls_losses,
                            loss=losses,
                            iou=ious,
                            sub_obj=sub_objs
                        )
                    )
                    recorder.print_speed((epoch - self.config.SEARCH.ARCH_EPOCH - 1) * len(val_loader) + iter + 1,
                                         batch_time.avg,
                                         (self.config.TRAIN.END_EPOCH - self.config.SEARCH.ARCH_EPOCH) * \
                                         len(val_loader), logger)

                    # write to tensorboard
                    writer = writer_dict['writer']
                    global_steps = writer_dict['train_global_steps']

                    writer.add_scalar('loss', loss, global_steps)
                    writer.add_scalar('cls_loss', cls_loss, global_steps)
                    writer.add_scalar('reg_loss', reg_loss, global_steps)
                    writer_dict['train_global_steps'] = global_steps + 1

        self.set_param_grad_state('Weights')
        for iter, batchinfo in enumerate(train_loader):
            data_time.update(time.time() - end)

            template = batchinfo['template'].to(self.device)  # bx3x127x127
            search = batchinfo['search'].to(self.device)  # bx3x255x255
            cls_label = batchinfo['cls_label'].type(torch.FloatTensor).to(self.device)  # bx31x31
            reg_label = batchinfo['reg_label'].float().to(self.device)  # bx31x31x4
            reg_weight = batchinfo['reg_weight'].float().to(self.device)  # bx31x31
            template_bbox = batchinfo['template_bbox'].to(self.device)  # bx4

            model_inputs = {'template': template,
                            'search': search,
                            'cls_label': cls_label,
                            'reg_label': reg_label,
                            'reg_weight': reg_weight,
                            'template_bbox': template_bbox,
                            }
            iou, loss, cls_loss, reg_loss, sub_obj = self.weight_step(model_inputs, model, search_stage)

            # record loss
            ious.update(iou, template.size(0))
            losses.update(loss, template.size(0))
            cls_losses.update(cls_loss, template.size(0))
            reg_losses.update(reg_loss, template.size(0))
            sub_objs.update(sub_obj, template.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

            if (iter + 1) % self.config.TRAIN.PRINT_FREQ == 0:
                logger.info(
                    '''Epoch: [{0}][{1}/{2}]\
                    lr: {lr:.7f}\
                    Batch Time: {batch_time.avg:.3f}s\
                    Data Time:{data_time.avg:.3f}s
                    REG_Loss:{reg_loss.avg:.5f}\
                    CLS_Loss:{cls_loss.avg:.5f}\
                    Loss:{loss.avg:.5f}\
                    IOU:{iou.avg:.5f}\
                    Sub_Obj:{sub_obj.avg:.5f}'''.format(
                        epoch,
                        iter + 1,
                        len(train_loader),
                        lr=cur_lr,
                        batch_time=batch_time,
                        data_time=data_time,
                        reg_loss=reg_losses,
                        cls_loss=cls_losses,
                        loss=losses,
                        iou=ious,
                        sub_obj=sub_objs
                        )
                )
                recorder.print_speed((epoch - 1) * len(train_loader) + iter + 1, batch_time.avg,
                                     self.config.TRAIN.END_EPOCH * len(train_loader), logger)

                # write to tensorboard
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']

                writer.add_scalar('loss', loss, global_steps)
                writer.add_scalar('cls_loss', cls_loss, global_steps)
                writer.add_scalar('reg_loss', reg_loss, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        return model, writer_dict

    def arch_step(self, inputs, model, search_stage):
        head_sampled_w_old, alpha_head_index = model.module.sample_branch('head', 2, search_stage=search_stage)
        stack_sampled_w_old, alpha_stack_index = model.module.sample_branch('stack', 2, search_stage=search_stage)

        dropped_model = DataParallel(self.Dropped_Network(model), device_ids=self.gpus).to(self.device)
        z, x, bbox = inputs['template'], inputs['search'], inputs['template_bbox']
        cls_pred, reg_pred, sub_obj = dropped_model(z, x, bbox)

        sub_obj = torch.mean(sub_obj)
        cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
        cls_loss = self.cls_criterion(cls_pred, cls_label)
        reg_loss, iou = self.reg_criterion(reg_pred, reg_label, reg_weight)
        cls_loss = torch.mean(cls_loss)
        reg_loss = torch.mean(reg_loss)
        iou = torch.mean(iou)
        loss = self.config.TRAIN.CLS_WEIGHT * cls_loss + self.config.TRAIN.REG_WEIGHT * reg_loss
        loss = torch.mean(loss)

        if self.config.SEARCH.IF_SUB_OBJ:
            loss_sub_obj = torch.log(sub_obj) / torch.log(torch.tensor(self.config.SEARCH.SUB_OBJ.LOG_BASE))
            sub_loss_factor = self.config.SEARCH.SUB_OBJ.SUB_LOSS_FACTOR
            loss += loss_sub_obj * sub_loss_factor

        self.arch_optimizer.zero_grad()
        loss.backward()

        if loader.is_valid_number(loss.item()):
            self.arch_optimizer.step()

        self.rescale_arch_params(head_sampled_w_old,
                                 stack_sampled_w_old,
                                 alpha_head_index,
                                 alpha_stack_index,
                                 model)
        return iou.detach().item(), loss.item(), cls_loss.item(), reg_loss.item(), sub_obj.item()

    def weight_step(self, inputs, model, search_stage):
        _, _ = model.module.sample_branch('head', self.weight_sample_num, search_stage=search_stage)
        _, _ = model.module.sample_branch('stack', self.weight_sample_num, search_stage=search_stage)

        dropped_model = DataParallel(self.Dropped_Network(model), device_ids=self.gpus).to(self.device)
        z, x, bbox = inputs['template'], inputs['search'], inputs['template_bbox']
        cls_pred, reg_pred, sub_obj = dropped_model(z, x, bbox)

        sub_obj = torch.mean(sub_obj)
        cls_label, reg_label, reg_weight = inputs['cls_label'], inputs['reg_label'], inputs['reg_weight']
        cls_loss = self.cls_criterion(cls_pred, cls_label)
        reg_loss, iou = self.reg_criterion(reg_pred, reg_label, reg_weight)
        cls_loss = torch.mean(cls_loss)
        reg_loss = torch.mean(reg_loss)
        iou = torch.mean(iou)
        loss = self.config.TRAIN.CLS_WEIGHT * cls_loss + self.config.TRAIN.REG_WEIGHT * reg_loss
        loss = torch.mean(loss)

        self.weight_optimizer.zero_grad()
        loss.backward()

        if self.config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm(model.parameters(), 10)  # gradient clip

        if loader.is_valid_number(loss.item()):
            self.weight_optimizer.step()

        return iou.detach().item(), loss.item(), cls_loss.item(), reg_loss.item(), sub_obj.item()

    def rescale_arch_params(self, alpha_head_weights_drop,
                            alpha_stack_weights_drop,
                            alpha_head_index,
                            alpha_stack_index,
                            model):

        def comp_rescale_value(old_weights, new_weights, index):
            old_exp_sum = old_weights.exp().sum()
            new_drop_arch_params = torch.gather(new_weights, dim=-1, index=index)
            new_exp_sum = new_drop_arch_params.exp().sum()
            rescale_value = torch.log(old_exp_sum / new_exp_sum)
            rescale_mat = torch.zeros_like(new_weights).scatter_(0, index, rescale_value.repeat(len(index)))
            return rescale_value, rescale_mat

        def rescale_params(old_weights, new_weights, indices):
            for i, (old_weights_block, indices_block) in enumerate(zip(old_weights, indices)):
                for j, (old_weights_branch, indices_branch) in enumerate(zip(old_weights_block, indices_block)):
                    rescale_value, rescale_mat = comp_rescale_value(old_weights_branch,
                                                                    new_weights[i][j],
                                                                    indices_branch)
                    new_weights[i][j].data.add_(rescale_mat)

        # rescale the arch params for head layers
        rescale_params(alpha_head_weights_drop, model.module.alpha_head_weights, alpha_head_index)
        # rescale the arch params for stack layers
        rescale_params(alpha_stack_weights_drop, model.module.alpha_stack_weights, alpha_stack_index)

    def set_param_grad_state(self, stage):
        def set_grad_state(params, state):
            for group in params:
                for param in group['params']:
                    param.requires_grad_(state)
        if stage == 'Arch':
            state_list = [True, False]  # [arch, weight]
        elif stage == 'Weights':
            state_list = [False, True]
        else:
            state_list = [False, False]
        set_grad_state(self.arch_optimizer.param_groups, state_list[0])
        set_grad_state(self.weight_optimizer.param_groups, state_list[1])
