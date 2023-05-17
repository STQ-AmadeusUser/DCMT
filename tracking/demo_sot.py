import _init_paths
import os
import cv2
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from os.path import exists, join, dirname, realpath
from matplotlib import pyplot as plt
import tracker.demo_tracker as tracker_builder
import utils.model_helper as loader
import utils.box_helper as boxhelper
import utils.log_helper as recorder
import utils.sot_builder as builder
import utils.read_file as reader
import search.model_derived as model_derived
from dataset.benchmark_loader import load_sot_benchmark as datafactory
from models.demosiaminference import DemoSiamInference
import pickle


def parse_args():
    parser = argparse.ArgumentParser(description='Test SOT trackers')
    parser.add_argument('--cfg', type=str, default='../experiments/UAV_s.yaml', help='yaml configure file name')
    parser.add_argument('--resume',  default=None, help='resume checkpoin, if None, use resume in config or epoch testing')
    parser.add_argument('--dataset',  default=None, help='evaluated benchmark, if None, use that in config')
    parser.add_argument('--arch_type', type=str, default='s', help='retrain densesiam from which type')
    parser.add_argument('--vis', default=False, type=bool, help='visualization')
    parser.add_argument('--video_path', default=None, help='whether run on a single video (.mp4 or others)')
    parser.add_argument('--video', default=None, type=str, help='eval one special video')

    args = parser.parse_args()

    return args


def track(inputs):
    siam_tracker = inputs['tracker']
    siam_net = inputs['network']
    video_info = inputs['video_info']
    args = inputs['args']
    config = inputs['config']
    writer, im_recorder = None, []
    start_frame, lost, toc, boxes, times = 0, 0, 0, [], []
    vis_path = os.path.join('./vis', config.TEST.DATA, video_info['name'])
    if not os.path.isdir(vis_path):
        os.makedirs(vis_path)

    image_files, gt = video_info['image_files'], video_info['gt']
    print('video length: ', len(gt))

    for f, image_file in enumerate(image_files):
        im = cv2.imread(image_file)
        if len(im.shape) == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

        tic = cv2.getTickCount()
        if f == start_frame:  # init
            # initialize video writer
            writer = cv2.VideoWriter('../video/demo_sot.mp4',
                                     cv2.VideoWriter_fourcc(*"mp4v"),
                                     30,
                                     (im.shape[1], im.shape[0]))
            if 'UAVTRACK112' in config.TEST.DATA or config.TEST.DATA == 'ANTIUAV' or config.TEST.DATA == 'UAVDARK135':
                init_rect = np.array(gt[f])
            else:
                init_rect = gt[f]
            cx, cy, w, h = boxhelper.get_axis_aligned_bbox(init_rect)  # center_x, center_y
            target_pos, target_sz = np.array([cx, cy]), np.array([w, h])

            init_inputs = {'image': im, 'pos': target_pos, 'sz': target_sz, 'model': siam_net}
            siam_tracker.init(init_inputs, vis_path)  # init tracker

            # location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
            boxes.append(1 if 'VOT' in config.TEST.DATA else init_rect)
            bbox = list(map(int, init_rect))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)

            writer.write(im)
            im_recorder.append(im)
            times.append(0.02)  # just used for testing on online saver which requires time recording, e.g. got10k

        elif f > start_frame:  # tracking
            try:
                gt_ = list(map(int, gt[f]))
                gt_ = [gt_[0], gt_[1], gt_[0] + gt_[2], gt_[1] + gt_[3]]
            except:
                gt_ = []
            state = siam_tracker.track(im, gt_, f)

            location = boxhelper.cxy_wh_2_rect(state['pos'], state['sz'])
            b_overlap = boxhelper.poly_iou(gt[f], location) if 'VOT' in config.TEST.DATA else 1
            times.append(0.02)
            if b_overlap > 0:
                boxes.append(location)
            else:
                boxes.append(2)
                start_frame = f + 5
                lost += 1

            bbox = list(map(int, location))
            cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 255), 3)
            im_recorder.append(im)
            writer.write(im)

            # TODO: cut down for long time tracking
            # if f > 1760:  # stop tracking when > 170
            #     break

        else:
            boxes.append(0)

        toc += cv2.getTickCount() - tic

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video_info['name'], toc, f / toc, lost))
    writer.release()
    for j, img in enumerate(im_recorder):
        cv2.imwrite('../video/vis_{}.png'.format(j), img)
        cv2.imwrite('../video/vis_{}.jpg'.format(j), img)
    boxes_, gts_ = [], []
    if 'gt_frames' in video_info.keys():
        for k, gt_id in enumerate(video_info['gt_frames']):
            try:
                boxes_.append(np.array(boxes)[gt_id])
                gts_.append(gt[k])
            except:
                break
        boxes = np.array(boxes_)
        gt = np.array(gts_)
    top_7 = plot_real_world(gt, boxes)
    for idx in top_7:
        if 'gt_frames' in video_info.keys():
            cle_img = im_recorder[video_info['gt_frames'][idx]]
            cv2.imwrite('../video/cle_{}.png'.format(video_info['gt_frames'][idx]), cle_img)
            cv2.imwrite('../video/cle_{}.jpg'.format(video_info['gt_frames'][idx]), cle_img)
        else:
            cle_img = im_recorder[idx]
            cv2.imwrite('../video/cle_{}.png'.format(idx), cle_img)
            cv2.imwrite('../video/cle_{}.jpg'.format(idx), cle_img)


def plot_real_world(gt, pred):
    len_video = len(pred)
    cle = []
    frames = list(range(len_video))
    for i in frames:
        gt_i = gt[i]
        pred_i = pred[i]
        gt_i_cx = gt_i[0] + gt_i[2] * 0.5
        gt_i_cy = gt_i[1] + gt_i[3] * 0.5
        pred_i_cx = pred_i[0] + pred_i[2] * 0.5
        pred_i_cy = pred_i[1] + pred_i[3] * 0.5
        cle_i = np.sqrt((gt_i_cx - pred_i_cx) ** 2 + (gt_i_cy - pred_i_cy) ** 2)
        cle.append(cle_i)
    plt.figure(figsize=(13, 2))
    plt.plot(frames, cle, label='CLE', c='Red', linewidth=3)
    plt.axhline(y=20, ls='--', c='green', linewidth=3)
    plt.savefig('../video/demo_cle.svg', format='svg')
    top_7 = list(np.argsort(cle)[-7:])
    top_7.reverse()
    print('Top-7 of cle: ', top_7)
    # plt.show()
    # print('haha')
    return top_7


def main():
    print('===> load config <====')
    args = parse_args()
    if args.cfg is not None:
        config = edict(reader.load_yaml(args.cfg))
    else:
        raise Exception('Please set the config file for tracking test!')

    # create derived model of super model
    with open('./arch/arch_{}.pkl'.format(str(args.arch_type)), 'rb') as f:
        arch_dict = pickle.load(f)
    arch_config = arch_dict['arch']
    derivedNetwork = getattr(model_derived, '%s_Net' % config.MODEL.NAME.upper())
    der_Net = lambda net_config: derivedNetwork(net_config, config=config)
    derived_model = der_Net(arch_config)

    # create model
    print('====> build model <====')
    siam_net = DemoSiamInference(derived_model, config)

    # load checkpoint
    print('===> init Siamese <====')
    if args.resume is None or args.resume == 'None':
        resume = config.TEST.RESUME
    else:
        resume = args.resume
    siam_net = loader.load_pretrain(siam_net, resume, print_unuse=False)
    siam_net.eval()
    siam_net = siam_net.cuda()

    # create tracker
    siam_tracker = tracker_builder.DemoTracker(config)

    # prepare video
    if args.dataset is None:
        dataset_loader = datafactory(config.TEST.DATA)
    else:
        config.TEST.DATA = args.dataset
        dataset_loader = datafactory(args.dataset)
    dataset = dataset_loader.load()
    video_keys = list(dataset.keys()).copy()

    for video in video_keys:
        if args.video is None or (args.video is not None and args.video == video):
            inputs = {'tracker': siam_tracker,
                      'network': siam_net,
                      'video_info': dataset[video],
                      'args': args,
                      'config': config,
                      }
            track(inputs)


if __name__ == '__main__':
    main()

