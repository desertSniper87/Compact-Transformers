#!/usr/bin/env python

import argparse
import os
import random
from time import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from PIL import Image

import src as models
from src.cct import _cct, CCT

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("_")
                     and callable(models.__dict__[name]))

DATASETS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'mean': [0.4914, 0.4822, 0.4465],
        'std': [0.2470, 0.2435, 0.2616]
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'mean': [0.5071, 0.4867, 0.4408],
        'std': [0.2675, 0.2565, 0.2761]
    },
    'icartoonface': {
        'num_classes': 5013,
        'img_size': 112,
        'mean': [0.5677, 0.5188, 0.4885],
        'std': [0.2040, 0.2908, 0.2848]
    }
}


def init_parser():
    parser = argparse.ArgumentParser(description='CIFAR10 quick evaluation script')

    # Data args
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')

    parser.add_argument('--dataset',
                        type=str.lower,
                        choices=['cifar10', 'cifar100', 'icartoonface'],
                        default='cifar10')

    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                        help='log frequency (by iteration)')

    parser.add_argument('--checkpoint-path',
                        type=str,
                        default='checkpoint.pth',
                        help='path to checkpoint (default: checkpoint.pth)')

    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N',
                        help='mini-batch size (default: 128)', dest='batch_size')

    parser.add_argument('-m', '--model',
                        type=str.lower,
                        choices=model_names,
                        default='cct_2', dest='model')

    parser.add_argument('-p', '--positional-embedding',
                        type=str.lower,
                        choices=['learnable', 'sine', 'none'],
                        default='learnable', dest='positional_embedding')

    parser.add_argument('--conv-layers', default=2, type=int,
                        help='number of convolutional layers (cct only)')

    parser.add_argument('--conv-size', default=3, type=int,
                        help='convolution kernel size (cct only)')

    parser.add_argument('--patch-size', default=4, type=int,
                        help='image patch size (vit and cvt only)')

    parser.add_argument('--gpu-id', default=0, type=int)

    parser.add_argument('--no-cuda', action='store_true',
                        help='disable cuda')

    parser.add_argument('--download', action='store_true',
                        help='download dataset (or verify if already downloaded)')

    return parser


def main():
    parser = init_parser()
    args = parser.parse_args()
    img_size = DATASETS[args.dataset]['img_size']
    num_classes = DATASETS[args.dataset]['num_classes']
    img_mean, img_std = DATASETS[args.dataset]['mean'], DATASETS[args.dataset]['std']



    model = _cct('cct_7_7x2_112', # Arch
                 True, # Pretrained
                 False, # Progress
                 7, # num_layers
                 4, # num_heads
                 2, # mlp_ratio
                 256, # embedded_dim
                 7, # kernel_size
                 None, # stride
                 None, # padding
                 'none', # positional_embedding
                 num_classes=5013,
                 img_size = 112,
                 patch_size = 7,
                 n_conv_layers=2,
                 bn_tf=False,
                 pretrained_weights='/run/media/torsho/87_portable/edu/dat/iCartoonFace/compact_transformer/output/train/20240211-203458-cct_7_7x2_112-112/model_best.pth.tar')

    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(999)
    random.seed(999)
    os.environ['CUDNN_DETERMINISTIC'] = '1'
    os.environ['PYTHONHASHSEED'] = str(0)

    normalize = [transforms.Normalize(mean=img_mean, std=img_std)]

    if (not args.no_cuda) and torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        model.cuda(args.gpu_id)

    # args.data ---> test_data for evaluation
    # The data is already resized we say

    transform_ = transforms.Compose([
            # transforms.Resize(img_size),
            transforms.ToTensor(),
            *normalize,
        ])

    model.eval()

    imgpaths = []
    imgpath_classids = []

    def get_predicted_class(img_path):
        img_path = f'/run/media/torsho/87_portable/edu/dat/iCartoonFace/personai_icartoonface_rectest/icartoonface_rectest/{img_path}'
        x = model(transform_(Image.open(img_path).convert('RGB')).unsqueeze(0))
        probabilities = nn.functional.softmax(x, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        return predicted_class.item()

    correct_num, total_num = 0, 0

    with open("/run/media/torsho/87_portable/edu/dat/iCartoonFace/evalution_code/rec_evalution_code/icartoonface_rectest_info.txt", 'r', encoding='utf-8') as f:
        for line in f.readlines():
            line_info = line.strip().split()
            if len(line_info) == 6:
                imgpaths.append(line_info[0])
                imgpath_classids.append(line_info[-1])
            if len(line_info) == 2:
                imgpath1, imgpath2 = line_info[0], line_info[1]
                idx1, idx2 = imgpaths.index(imgpath1), imgpaths.index(imgpath2)
                total_num += 1
                if get_predicted_class(imgpath1) == get_predicted_class(imgpath2) \
                        and imgpath_classids[idx1] == imgpath_classids[idx2]:
                    print(f'ok match' , end=' ')
                    correct_num += 1
                elif imgpath_classids[idx1] == -1 or imgpath_classids[idx2] == -1:
                    print(f'ok no match' , end=' ')
                    correct_num += 1
                elif get_predicted_class(imgpath1) != get_predicted_class(imgpath2) \
                    and imgpath_classids[idx1] != imgpath_classids[idx2]:
                    print(f'ok no match' , end=' ')
                    correct_num += 1
                else:
                    print(f'not ok' , end=' ')
                print(f'\t{idx1}\t{idx2}\t{imgpath1}\t{imgpath_classids[idx1]}\t{get_predicted_class(imgpath1)}\t{imgpath2}\t{imgpath_classids[idx2]}\t{get_predicted_class(imgpath2)}')

    print(100.0 * correct_num / total_num)


if __name__ == '__main__':
    main()
