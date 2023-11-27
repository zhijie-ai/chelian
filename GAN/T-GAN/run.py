#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) Tencent Inc.
# Author: ranchodzu (ranchodzu@tencent.com)
# TIME: 2023/1/9 10:01
# -*- coding: utf-8 -*-
# @Time    : 2022/3/10
# @Author  : shrekyuan
# @File    : run.py

import argparse
import os
import sys

#sys.path.append("/apdcephfs_cq2/share_919031/qingyuliu/pytorch/audio-textual-attbilstm-git/")

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--experiment',default='', type=str)
  parser.add_argument('--dataset',default='', type=str)
  parser.add_argument("--batch_size", default=8, type=int)
  parser.add_argument('--checkpoint', default='null', type=str, help='saving pretrained model')

  args_ = parser.parse_args()
  return args_

args = parse_args()
cmd_str = "python  /apdcephfs_cq2/share_919031/qingyuliu/pytorch/audio-textual-attbilstm-git/train.py"
#cmd_str = "python  /data/cfs/qingyuliu/pytorch/audio-textual-attbilstm-git/train.py"
cmd_str = cmd_str + " --experiment " + str(args.experiment)  + " --dataset " + str(args.dataset) + " --batch_size " + str(args.batch_size)+ " --checkpoint " + str(args.checkpoint)

print(cmd_str)
####挂载cfs####
#os.system("mkdir -p /data/cfs/")
#os.system("sudo mount -t nfs -o vers=3,nolock,noresvport $cfs_ip:/$cfs_id /data/cfs/")
#os.system("ls /data/cfs/")
########
os.system("pip install regex")
os.system("pip install boto3")

os.system(cmd_str)
