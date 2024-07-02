# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/6/28 15:11
# @User   : RANCHODZU
# @File   : options.py
# @e-mail : zushoujie@ghgame.cn
import argparse
from datetime import datetime
import torch


def get_options():
    current_time = datetime.now()
    # t = current_time.strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser(description="Args for  model")
    time = current_time.strftime('%Y-%m-%d-%H-%M')
    parser.add_argument('--current_time', default=time, type=str)
    parser.add_argument('--n_steps', default=1000, type=int)
    parser.add_argument('--n_samples', default=16, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_channels', default=1, type=int)
    parser.add_argument('--epochs', default=40, type=int)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--image_size', default=32, type=float)
    parser.add_argument('--local_rank', default=0, type=float)
    parser.add_argument('--tensorboard_path', default='tensorboards', type=str)
    parser.add_argument('--tensorboard_name', default='ddpm-{}'.format(time), type=str)
    parser.add_argument('--model_dir', default='models/ddpm-{}'.format(time), type=str)
    parser.add_argument('--sample_dir', default='samples'.format(time), type=str)
    parser.add_argument('--gradient_accumulation_step', default=10, type=int)
    parser.add_argument('--device', default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), type=str)

    cmd_args = parser.parse_args()
    return cmd_args

