# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/6/28 17:59
# @User   : RANCHODZU
# @File   : topylogy.py
# @e-mail : zushoujie@ghgame.cn

from options import get_options
from model import DenoiseDiffusion
from unet import UNet
from torchsummary import summary
import torch

args = get_options()

unet = UNet(args.image_channels).to(args.device)
ddpm = DenoiseDiffusion(unet, args.n_steps, device=args.device)

x = torch.randn(1, 32, 32).to(args.device)
# t = torch.randint(0, 100, (1)).to(args.device)
print(summary(unet, (4, ), batch_size=6))

