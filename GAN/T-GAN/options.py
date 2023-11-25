# ----------------------------------------------
# -*- encoding=utf-8 -*-                      #
# __author__:'焉知飞鱼'                        #
# CreateTime:                                 #
#       2023/2/21 10:33                       #
#                                             #
#               天下风云出我辈，                 #
#               一入江湖岁月催。                 #
#               皇图霸业谈笑中，                 #
#               不胜人生一场醉。                 #
# ----------------------------------------------
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

import argparse


def get_options(args=None):
    parser = argparse.ArgumentParser(
        description="Args for  model")

    parser.add_argument('--l2_w', type=float, default='0.0', help='weight decay')
    parser.add_argument('--epochs', type=int, default='10000', help='')
    parser.add_argument('--batch_size', type=int, default='64', help='')#
    parser.add_argument('--iters_per_sample', type=int, default='50')
    parser.add_argument('--d_lr', type=float, default='0.00001', help='dis learning rate')
    parser.add_argument('--g_lr', type=float, default='0.0001', help='gen learning rate')

    opts = parser.parse_args(args)

    return opts


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
