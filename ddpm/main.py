import torchvision.transforms as trans
import torchvision
import torch
from torch.utils.data import DataLoader
from unet import UNet
from dataset import DdpmDataset
from model import DenoiseDiffusion
from tqdm import trange, tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from trainer import Trainer
import logging
from options import get_options


def main(args, logger):
    transformer = trans.Compose([trans.Pad(2),
                                 # trans.Resize(size=(_args.image_size, _args.image_size)),
                                 trans.ToTensor()])
    minist = torchvision.datasets.MNIST('./data', download=True, train=True,
                                        transform=transformer)
    logger.info('{} {}'.format(minist.data[0].shape, minist.data[0].dtype))
    logger.info('torch.cuda.is_available(): {}'.format(torch.cuda.is_available()))
    minist = DdpmDataset(minist.data, minist.targets, args.device)
    # print(minist.X.device)
    # T = trans.Compose([trans.ToTensor()])
    dataloader = DataLoader(minist, batch_size=args.batch_size, shuffle=True, drop_last=True)

    unet = UNet(args.image_channels).to(args.device)
    ddpm = DenoiseDiffusion(unet, args.n_steps, device=args.device)
    trainer = Trainer(ddpm, dataloader, logger, args)
    trainer.train()
    logger.info('train process done!!!!!!!!!!!!!!!!!')

def get_logger():
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s-%(filename)s-%(levelname)s-line %(lineno)d - %(message)s')

    handler = logging.FileHandler('logs/log.txt')
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    log.addHandler(handler)
    log.addHandler(console)
    log.setLevel(logging.INFO)

    return log


if __name__ == '__main__':
    cmd_args = get_options()

    logger = get_logger()

    cmd_args.logger = logger
    logger.info('current_time:{}'.format(cmd_args.current_time))
    logger.info('>>>>>>>>>>>>>>>>>{}<<<<<<<<<<<<<<<<'.format(cmd_args))

    main(cmd_args, logger)
