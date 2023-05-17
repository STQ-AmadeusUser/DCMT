from __future__ import division

import os
import cv2
import json
import math
import random
import logging
import numpy as np
import torchvision.transforms as transforms
from scipy.ndimage.filters import gaussian_filter
from os.path import join
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

import utils.box_helper as boxhelper
import utils.augmentation as auger

import sys
sys.path.append('../')


sample_random = random.Random()


class SiameseDataset(Dataset):
    def __init__(self, cfg):
        super(SiameseDataset, self).__init__()
        # pair information
        self.template_size = cfg.TRAIN.TEMPLATE_SIZE
        self.search_size = cfg.TRAIN.SEARCH_SIZE
        self.stride = cfg.MODEL.STRIDE
        self.cfg = cfg
        self.score_size = 31

        # data augmentation
        self.color = cfg.TRAIN.DATASET.AUG.COMMON.COLOR
        self.flip = cfg.TRAIN.DATASET.AUG.COMMON.FLIP
        self.rotation = cfg.TRAIN.DATASET.AUG.COMMON.ROTATION
        self.blur = cfg.TRAIN.DATASET.AUG.COMMON.BLUR
        self.gray = cfg.TRAIN.DATASET.AUG.COMMON.GRAY
        self.label_smooth = cfg.TRAIN.DATASET.AUG.COMMON.LABELSMOOTH
        self.mixup = cfg.TRAIN.DATASET.AUG.COMMON.MIXUP
        self.neg = cfg.TRAIN.DATASET.AUG.COMMON.NEG
        self.jitter = None

        self.shift_s = cfg.TRAIN.DATASET.AUG.SEARCH.SHIFTs
        self.scale_s = cfg.TRAIN.DATASET.AUG.SEARCH.SCALEs
        self.shift_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SHIFT
        self.scale_e = cfg.TRAIN.DATASET.AUG.EXEMPLAR.SCALE

        # grids on input image
        self.grids()

        self.transform_extra = transforms.Compose(
            [transforms.ToPILImage(), ] +
            ([transforms.ColorJitter(0.05, 0.05, 0.05, 0.05), ] if self.color > random.random() else [])
            + ([transforms.RandomHorizontalFlip(), ] if self.flip > random.random() else [])
            + ([transforms.RandomRotation(degrees=10), ] if self.rotation > random.random() else [])
            + ([transforms.Grayscale(num_output_channels=3), ] if self.gray > random.random() else [])
        )

        # train data information
        print('train datas: {}'.format(cfg.TRAIN.DATASET.WHICH_USE))
        self.train_datas = []  # all train dataset
        start = 0
        self.num = 0
        for data_name in cfg.TRAIN.DATASET.WHICH_USE:
            dataset = subData(cfg, data_name, start)
            self.train_datas.append(dataset)
            start += dataset.num  # real video number
            self.num += dataset.num_use  # the number used for subset shuffle

        videos_per_epoch = cfg.TRAIN.DATASET.VIDEOS_PER_EPOCH
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self._shuffle()
        print(cfg)

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        pick a vodeo/frame --> pairs --> data aug --> label
        """
        index = self.pick[index]
        dataset, index = self._choose_dataset(index)

        if random.random() < self.neg:  # neg
            if dataset.data_name in ['VISDRONE_DET', 'VISDRONE_VID', 'PURDUE']:
                template, search = dataset._get_hard_negative_target(index, dataset.data_name)
                if template is None and search is None:  # get hard neg samples failed
                    template = dataset._get_negative_target(index)
                    search = np.random.choice(self.train_datas)._get_negative_target()
            else:
                template = dataset._get_negative_target(index)
                search = np.random.choice(self.train_datas)._get_negative_target()
            neg = True
        else:
            template, search = dataset._get_pairs(index, dataset.data_name)
            neg = False

        template, search = self.check_exists(index, dataset, template, search)
        template, search = self.check_damaged(index, dataset, template, search)

        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])

        template_box = self._toBBox(template_image, template[1])
        search_box = self._toBBox(search_image, search[1])

        # plt.figure()
        # cv2.rectangle(template_image, (int(template_box.x1), int(template_box.y1)), (int(template_box.x2), int(template_box.y2)), color=(255, 0, 0), thickness=3)
        # plt.imshow(template_image)
        # plt.show()
        # cv2.rectangle(search_image, (int(search_box.x1), int(search_box.y1)), (int(search_box.x2), int(search_box.y2)), color=(255, 0, 0), thickness=3)
        # plt.imshow(search_image)
        # plt.show()

        template, bbox_t, dag_param_t = self._augmentation(template_image, template_box, self.template_size)
        search, bbox, dag_param = self._augmentation(search_image, search_box, self.search_size, search=True)

        # from PIL image to numpy
        template = np.array(template)
        search = np.array(search)

        if neg:
            cls_label = np.zeros((self.score_size, self.score_size))
        else:
            cls_label = self._dynamic_label([self.score_size, self.score_size], dag_param.shift)

        reg_label, reg_weight = self.reg_label(bbox)

        template_mask = self.te_mask(bbox_t)
        jitterBox, jitter_ious = self.jitter_box(bbox)

        template, search = map(lambda x: np.transpose(x, (2, 0, 1)).astype(np.float32), [template, search])

        outputs = {'template': template,
                   'search': search,
                   'cls_label': cls_label,
                   'reg_label': reg_label,
                   'reg_weight': reg_weight,
                   'template_bbox': np.array(bbox_t, np.float32),
                   'search_bbox': np.array(bbox, np.float32),
                   'template_mask': template_mask,
                   'jitterBox': jitterBox,
                   'jitter_ious': jitter_ious,
                   }

        outputs = self.data_package(outputs)

        return outputs

    def data_package(self, outputs):
        clean = []
        for k, v in outputs.items():
            if v is None:
                clean.append(k)
        if len(clean) == 0:
            return outputs
        else:
            for k in clean:
                del outputs[k]

        return outputs

    def check_exists(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if not (os.path.exists(template[0]) and os.path.exists(search[0])):
                index = random.randint(0, 100)
                template, search = dataset._get_pairs(index, name)
                continue
            else:
                return template, search

    def check_damaged(self, index, dataset, template, search):
        name = dataset.data_name
        while True:
            if cv2.imread(template[0]) is None or cv2.imread(search[0]) is None:
                index = random.randint(0, 100)
                template, search = dataset._get_pairs(index, name)
                continue
            else:
                return template, search

    def _shuffle(self):
        """
        random shuffel
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for subset in self.train_datas:
                sub_p = subset.pick
                p += sub_p
            sample_random.shuffle(p)

            pick += p
            m = len(pick)
        self.pick = pick
        print("dataset length {}".format(self.num))

    def _choose_dataset(self, index):
        for dataset in self.train_datas:
            if dataset.start + dataset.num > index:
                return dataset, index - dataset.start

    def _posNegRandom(self):
        """
        random number from [-1, 1]
        """
        return random.random() * 2 - 1.0

    def _toBBox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = self.template_size

        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = boxhelper.center2corner(boxhelper.Center(cx, cy, w, h))
        return bbox

    def _crop_hwc(self, image, bbox, out_sz, padding=(0, 0, 0)):
        """
        crop image
        """
        bbox = [float(x) for x in bbox]
        a = (out_sz - 1) / (bbox[2] - bbox[0])
        b = (out_sz - 1) / (bbox[3] - bbox[1])
        c = -a * bbox[0]
        d = -b * bbox[1]
        mapping = np.array([[a, 0, c],
                            [0, b, d]]).astype(np.float)
        crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
        return crop

    def _draw(self, image, box, name):
        """
        draw image for debugging
        """
        draw_image = np.array(image.copy())
        x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
        cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.circle(draw_image, (int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)), 3, (0, 0, 255))
        cv2.putText(draw_image, '[x: {}, y: {}]'.format(int(round(x1 + x2) / 2), int(round(y1 + y2) / 2)),
                    (int(round(x1 + x2) / 2) - 3, int(round(y1 + y2) / 2) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (255, 255, 255), 1)
        cv2.imwrite(name, draw_image)

    def _draw_reg(self, image, grid_x, grid_y, reg_label, reg_weight, save_path, index):
        """
        visiualization
        reg_label: [l, t, r, b]
        """
        draw_image = image.copy()
        # count = 0
        save_name = join(save_path, '{:06d}.jpg'.format(index))
        h, w = reg_weight.shape
        for i in range(h):
            for j in range(w):
                if not reg_weight[i, j] > 0:
                    continue
                else:
                    x1 = int(grid_x[i, j] - reg_label[i, j, 0])
                    y1 = int(grid_y[i, j] - reg_label[i, j, 1])
                    x2 = int(grid_x[i, j] + reg_label[i, j, 2])
                    y2 = int(grid_y[i, j] + reg_label[i, j, 3])

                    draw_image = cv2.rectangle(draw_image, (x1, y1), (x2, y2), (0, 255, 0))

        cv2.imwrite(save_name, draw_image)

    def _mixupRandom(self):
        """
        gaussian random -- 0.3~0.7
        """
        return random.random() * 0.4 + 0.3

    # ------------------------------------
    # function for data augmentation
    # ------------------------------------
    def _augmentation(self, image, bbox, size, search=False):
        """
        data augmentation for input pairs
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        if search:
            param.shift = (self._posNegRandom() * self.shift_s, self._posNegRandom() * self.shift_s)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_s), (1.0 + self._posNegRandom() * self.scale_s))  # scale change
        else:
            param.shift = (self._posNegRandom() * self.shift_e, self._posNegRandom() * self.shift_e)  # shift
            param.scale = (
            (1.0 + self._posNegRandom() * self.scale_e), (1.0 + self._posNegRandom() * self.scale_e))  # scale change

        crop_bbox, _ = auger.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        x1, y1 = crop_bbox.x1, crop_bbox.y1
        bbox = boxhelper.BBox(bbox.x1 - x1, bbox.y1 - y1, bbox.x2 - x1, bbox.y2 - y1)

        scale_x, scale_y = param.scale
        bbox = boxhelper.Corner(bbox.x1 / scale_x, bbox.y1 / scale_y, bbox.x2 / scale_x, bbox.y2 / scale_y)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        if self.blur > random.random():
            image = gaussian_filter(image, sigma=(1, 1, 0))

        image = self.transform_extra(image)  # other data augmentation
        return image, bbox, param

    def _mixupShift(self, image, size):
        """
        random shift mixed-up image
        """
        shape = image.shape
        crop_bbox = boxhelper.center2corner((shape[0] // 2, shape[1] // 2, size, size))
        param = edict()

        param.shift = (self._posNegRandom() * 64, self._posNegRandom() * 64)  # shift
        crop_bbox, _ = boxhelper.aug_apply(boxhelper.Corner(*crop_bbox), param, shape)

        image = self._crop_hwc(image, crop_bbox, size)  # shift and scale

        return image

    # ------------------------------------
    # function for creating training label
    # ------------------------------------
    def _dynamic_label(self, fixedLabelSize, c_shift, rPos=2, rNeg=0):
        if isinstance(fixedLabelSize, int):
            fixedLabelSize = [fixedLabelSize, fixedLabelSize]

        assert (fixedLabelSize[0] % 2 == 1)

        d_label = self._create_dynamic_logisticloss_label(fixedLabelSize, c_shift, rPos, rNeg)

        return d_label

    def _create_dynamic_logisticloss_label(self, label_size, c_shift, rPos=2, rNeg=0):
        if isinstance(label_size, int):
            sz = label_size
        else:
            sz = label_size[0]

        sz_x = sz // 2 + int(-c_shift[0] / 8)  # 8 is strides
        sz_y = sz // 2 + int(-c_shift[1] / 8)

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        dist_to_center = np.abs(x) + np.abs(y)  # Block metric
        label = np.where(dist_to_center <= rPos,
                         np.ones_like(y),
                         np.where(dist_to_center < rNeg,
                                  0.5 * np.ones_like(y),
                                  np.zeros_like(y)))
        return label

    def grids(self):
        """
        each element of feature map on input search image
        :return: H*W*2 (position for each element)
        """
        sz = self.score_size

        sz_x = sz // 2
        sz_y = sz // 2

        x, y = np.meshgrid(np.arange(0, sz) - np.floor(float(sz_x)),
                           np.arange(0, sz) - np.floor(float(sz_y)))

        self.grid_to_search = {}
        self.grid_to_search_x = x * self.stride + self.search_size // 2
        self.grid_to_search_y = y * self.stride + self.search_size // 2

    def reg_label(self, bbox):
        """
        generate regression label
        :param bbox: [x1, y1, x2, y2]
        :return: [l, t, r, b]
        """
        x1, y1, x2, y2 = bbox
        l = self.grid_to_search_x - x1  # [17, 17]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)  # [17, 17, 4]
        reg_label_min = np.min(reg_label, axis=-1)
        inds_nonzero = (reg_label_min > 0).astype(float)

        return reg_label, inds_nonzero

    def cls_reg_label(self, bbox):
        x1, y1, x2, y2 = bbox
        half_w, half_h = (x2 - x1) / 2, (y2 - y1) / 2

        l = self.grid_to_search_x - x1  # [21, 21]
        t = self.grid_to_search_y - y1
        r = x2 - self.grid_to_search_x
        b = y2 - self.grid_to_search_y

        # classic wrong code, please do not de-comment
        # ignore_l = (l > 0.2 * half_w) & (l <= 0.5 * half_w)
        # ignore_r = (r > 0.2 * half_w) & (r <= 0.5 * half_w)
        # ignore_t = (t > 0.2 * half_h) & (t <= 0.5 * half_h)
        # ignore_b = (b > 0.2 * half_h) & (b <= 0.5 * half_h)
        ignore_l = (l > 0.2 * half_w)
        ignore_r = (r > 0.2 * half_w)
        ignore_t = (t > 0.2 * half_h)
        ignore_b = (b > 0.2 * half_h)
        neg_inbox = ignore_l * ignore_t * ignore_r * ignore_b
        # cls_inds_neg_inbox = np.asarray(neg_inbox, dtype=np.float64)
        cls_inds_neg_inbox = np.unravel_index(np.where(neg_inbox == 1), neg_inbox.shape)[1]

        trust_l = l > 0.5 * half_w
        trust_r = r > 0.5 * half_w
        trust_t = t > 0.5 * half_h
        trust_b = b > 0.5 * half_h
        pos_inbox = trust_l * trust_t * trust_r * trust_b
        # cls_inds_pos_inbox = np.asarray(pos_inbox, dtype=np.float64)
        cls_inds_pos_inbox = np.unravel_index(np.where(pos_inbox == 1), pos_inbox.shape)[1]

        l, t, r, b = map(lambda x: np.expand_dims(x, axis=-1), [l, t, r, b])
        reg_label = np.concatenate((l, t, r, b), axis=-1)  # [21, 21, 4]
        reg_label_min = np.min(reg_label, axis=-1)
        reg_inds_inbox = (reg_label_min > 0).astype(float)

        return reg_label, reg_inds_inbox, cls_inds_neg_inbox, cls_inds_pos_inbox

    def te_mask(self, bbox):
        """
        generate mask for template frame
        :param bbox: [x1, y1, x2, y2]
        :return: binary mask
        """
        x1, y1, x2, y2 = bbox
        mask = np.zeros((self.template_size, self.template_size))
        r_start, r_end = int(y1), math.ceil(y2 + 1)
        c_start, c_end = int(x1), math.ceil(x2 + 1)

        mask[r_start:r_end, c_start:c_end] = 1

        return mask

    def IOUgroup(self, boxes, gt_xyxy):
        # overlap

        x1, y1, x2, y2 = gt_xyxy.reshape(4, )
        pred_x1, pred_y1, pred_x2, pred_y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
        xx1 = np.maximum(pred_x1, x1)  # 17*17
        yy1 = np.maximum(pred_y1, y1)
        xx2 = np.minimum(pred_x2, x2)
        yy2 = np.minimum(pred_y2, y2)

        ww = np.maximum(0, xx2 - xx1)
        hh = np.maximum(0, yy2 - yy1)

        ww[ww < 0] = 0
        hh[hh < 0] = 0

        area = (x2 - x1) * (y2 - y1)

        target_a = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)

        inter = ww * hh
        overlap = inter / (np.abs(area + target_a - inter) + 1)

        overlap[overlap > 1] = 0

        return overlap

    def jitter_box(self, box):
        """
        :param box: [x1, y1, x2, y2] 1*4
        :return:
        """

        box = np.array([box.x1, box.y1, box.x2, box.y2]).reshape(1, 4)
        box_rep = box.repeat(96, axis=0)

        add = np.array([4, 8, 12, 16]).astype(np.float)
        minus = -1 * add
        add2 = add.reshape(4, 1).repeat(2, axis=-1)
        minus2 = minus.reshape(4, 1).repeat(2, axis=1)

        if self.jitter is None:
            shift = np.zeros((96, 4))

            # settle (x1, y1) change (x2, y2)
            shift[0:4, 2] += add
            shift[4:8, 2] += minus
            shift[8:12, 3] += add
            shift[12:16, 3] += minus
            shift[16:20, 2:4] += add2
            shift[20:24, 2:4] += minus2

            # settle (x2, y1) change (x1, y2)
            shift[24:28, 0] += add
            shift[28:32, 0] += minus
            shift[32:36, 3] += add
            shift[36:40, 3] += minus
            shift[40:44, 0] += add
            shift[40:44, 3] += add
            shift[44:48, 0] += minus
            shift[44:48, 3] += minus

            # settle (x2, y2) change (x1, y1)
            shift[48:52, 0] += add
            shift[52:56, 0] += minus
            shift[56:60, 1] += add
            shift[60:64, 1] += minus
            shift[64:68, 0:2] += add2
            shift[68:72, 0:2] += minus2

            # settle (x1, y2) change (x2, y1)
            shift[72:76, 2] += add
            shift[76:80, 2] += minus
            shift[80:84, 1] += add
            shift[84:88, 1] += minus
            shift[88:92, 1:3] += add2
            shift[92:96, 1:3] += minus2

            self.jitter = shift

        jitter_box = box_rep + self.jitter
        jitter_box = np.clip(jitter_box, 0, 255)

        # ious:
        ious = self.IOUgroup(jitter_box, box)

        return jitter_box, ious


# ---------------------
# for a single dataset
# ---------------------
class subData(object):
    """
    for training with multi dataset
    """

    def __init__(self, cfg, data_name, start):
        self.data_name = data_name
        self.start = start

        info = cfg.TRAIN.DATASET.CONFIG[data_name]
        self.frame_range = info.RANGE
        self.num_use = info.USE
        self.root = info.PATH

        with open(info.ANNOTATION) as fin:
            self.labels = json.load(fin)
            self._clean()
            self.num = len(self.labels)  # video numer

        self._shuffle()

    def _clean(self):
        """
        remove empty videos/frames/annos in dataset
        """
        # no frames
        to_del = []
        for video in self.labels:
            for track in self.labels[video]:
                frames = self.labels[video][track]
                frames = list(map(int, frames.keys()))
                frames.sort()
                self.labels[video][track]['frames'] = frames
                if len(frames) <= 0:
                    print("warning {}/{} has no frames.".format(video, track))
                    to_del.append((video, track))

        for video, track in to_del:
            try:
                del self.labels[video][track]
            except:
                pass

        # no track/annos
        to_del = []

        if self.data_name == 'YTB':
            to_del.append('train/1/YyE0clBPamU')  # This video has no bounding box.
        print(self.data_name)

        for video in self.labels:
            if len(self.labels[video]) <= 0:
                print("warning {} has no tracks".format(video))
                to_del.append(video)

        for video in to_del:
            try:
                del self.labels[video]
            except:
                pass

        self.videos = list(self.labels.keys())
        print('{} loaded.'.format(self.data_name))

    def _shuffle(self):
        """
        shuffel to get random pairs index (video)
        """
        lists = list(range(self.start, self.start + self.num))
        m = 0
        pick = []
        while m < self.num_use:
            sample_random.shuffle(lists)
            pick += lists
            m += self.num

        self.pick = pick[:self.num_use]
        return self.pick

    def _get_image_anno(self, video, track, frame):
        """
        get image and annotation
        """

        frame = "{:06d}".format(frame)

        image_path = join(self.root, video, "{}.{}.x.jpg".format(frame, track))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def _get_pairs(self, index, data_name):
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]
        try:
            track_info.pop('frames')
        except KeyError:
            pass
        frames = list(track_info.keys())

        template_frame = random.randint(0, len(frames) - 1)

        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]

        if data_name in ['VISDRONE_DET', 'VISDRONE_VID', 'PURDUE']:
            search_frame = template_frame
            template_frame = int(frames[template_frame])
            search_frame = int(frames[search_frame])
        else:
            template_frame = int(frames[template_frame])
            search_frame = int(random.choice(search_range))

        return self._get_image_anno(video_name, track, template_frame), \
               self._get_image_anno(video_name, track, search_frame)

    def _get_negative_target(self, index=-1):
        """
        dasiam neg
        """
        if index == -1:
            index = random.randint(0, self.num - 1)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = random.choice(list(video.keys()))
        track_info = video[track]

        try:
            track_info.pop('frames')
        except KeyError:
            pass
        frames = list(track_info.keys())
        frame = int(random.choice(frames))

        return self._get_image_anno(video_name, track, frame)

    def _get_hard_negative_target(self, index, data_name):  # for VISDRONE_DET, VISDRONE_VID
        """
        get training pairs
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track_list = list(video.keys())

        if data_name == 'VISDRONE_DET':
            track = random.choice(track_list)
            track_info = video[track]
            try:
                track_info.pop('frames')
            except KeyError:
                pass
            frames = list(track_info.keys())

            template_frame = random.randint(0, len(frames) - 1)

            left = max(template_frame - self.frame_range, 0)
            right = min(template_frame + self.frame_range, len(frames) - 1) + 1
            search_range = frames[left:right]

            template_frame = int(frames[template_frame])

            if template_frame == int(search_range[0]) and template_frame == int(search_range[-1]):
                return None, None

            while True:
                search_frame = int(random.choice(search_range))
                if template_frame != search_frame:
                    break

            return self._get_image_anno(video_name, track, template_frame), \
                   self._get_image_anno(video_name, track, search_frame)

        elif data_name in ['VISDRONE_VID', 'PURDUE']:
            if len(track_list) <= 1:
                return None, None
            template_track_id = random.randint(0, len(track_list) - 1)
            template_track = track_list[template_track_id]

            left = max(template_track_id - self.frame_range, 0)
            right = min(template_track_id + self.frame_range, len(track_list) - 1) + 1
            search_range = track_list[left:right]
            while True:
                search_track = random.choice(search_range)
                if template_track != search_track:
                    break
            # template sample
            template_track_info = video[template_track]
            try:
                template_track_info.pop('frames')
            except KeyError:
                pass
            template_frame = random.choice(list(template_track_info.keys()))
            # search sample
            search_track_info = video[search_track]
            try:
                search_track_info.pop('frames')
            except KeyError:
                pass
            search_frame = random.choice(list(search_track_info.keys()))

            return self._get_image_anno(video_name, template_track, int(template_frame)), \
                   self._get_image_anno(video_name, search_track, int(search_frame))

        else:
            raise NotImplementedError
