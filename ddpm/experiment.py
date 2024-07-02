# !/user/bin/python
# coding  : utf-8
# @Time   : 2024/6/28 15:00
# @User   : RANCHODZU
# @File   : experiment.py
# @e-mail : zushoujie@ghgame.cn

import torch
from options import get_options
import matplotlib.pyplot as plt

cmd_args = get_options()

model = torch.load('models/ddpm-2024-06-28-11-34/ddpm.bin')
# print(model.eps_model.state_dict())

device = torch.device('cuda')
x = torch.randn([2, cmd_args.image_channels,
                 cmd_args.image_size, cmd_args.image_size], device=cmd_args.device)
t = 1
tmp = x.new_full((2,), t).to(torch.int64)
x = model.p_sample(x, tmp)
x = x.permute(0, 2, 3, 1)
fig, ax = plt.subplots(1, 2)
# plt.imshow(x[0, :, :, 0].cpu().detach().numpy(), cmap='gray')
ax[0].imshow(x[0, :, :, 0].cpu().detach().numpy(), cmap='gray')
ax[1].imshow(x[0, :, :, 0].cpu().detach().numpy(), cmap='gray')
plt.tight_layout()
plt.show()
