# coding=gbk
#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# import _init_paths
from model.config import cfg
from model.test import im_detect
from torchvision.ops import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import random
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1
import torch

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train',
           'tvmonitor')

def demo(net):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im = cv2.imread('004545.jpg')
    # im=cv2.resize(im,None,None,fx=2,fy=2)
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(
        timer.total_time(), boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    Colors = [[random.randint(0, 256) for _ in range(3)] for _ in CLASSES]
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(
            torch.from_numpy(cls_boxes), torch.from_numpy(cls_scores),
            NMS_THRESH)
        dets = dets[keep.numpy(), :]

        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        for i in inds:
            bbox = dets[i, :4]
            score = dets[i, -1]
            cv2.rectangle(im,(int(bbox[0]),int(bbox[1])),(int(bbox[2]),int(bbox[3])),Colors[CLASSES.index(cls)],2)
            cv2.putText(im,'{}:{:.2f}'.format(str(cls),score),(int(bbox[0]),int(bbox[1])+12),1,1.2,Colors[CLASSES.index(cls)],2)
    cv2.imshow('11',im)
    cv2.waitKey(0)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net',
        dest='demo_net',
        help='Network to use [vgg16 res101 mobile]',
        default='res101')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net

    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    elif demonet == 'mobile':
        net = mobilenetv1()
    else:
        raise NotImplementedError
    net.create_architecture(21, tag='default', anchor_scales=[8, 16, 32])

    net.load_state_dict(
        torch.load('model/res101_faster_rcnn_iter_110000.pth',map_location=torch.device('cpu')))

    net.eval()
    if not torch.cuda.is_available():
        net._device = 'cpu'
    net.to(net._device)


    demo(net)


