import torch
import torchvision.utils
from tqdm import trange, tqdm
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import save_image, make_grid  # save_image默认会调用make_grid方法

'''
1. 正对imgs = torch.randn(16,3,32,32)这种批量图片，可以直接调用save_image保存
2. save_image+make_grid的组合好像有问题
'''
class Trainer:
    def __init__(self, ddpm, dataloader, logger, args):
        self.model = ddpm
        self.dataloader = dataloader
        self.image_channels = args.image_channels
        self.image_size = args.image_size
        self.learning_rate = args.learning_rate
        self.device = args.device
        self.n_samples = args.n_samples
        self.epochs = args.epochs
        self.n_steps = args.n_steps
        self.model_dir = args.model_dir
        self.gradient_accumulation_step = args.gradient_accumulation_step
        self.logger = logger
        self.summary_writer = SummaryWriter(
            os.path.join(args.tensorboard_path,
                         str(datetime.now().strftime("%Y-%m-%d_%H_%M_") + args.tensorboard_name)))

    def train(self):
        global_step = 0
        optimizer = torch.optim.Adam(self.model.eps_model.parameters(), lr=self.learning_rate)
        for ep in trange(self.epochs, desc='Epoch'):
            data_loader = tqdm(self.dataloader, desc='Iteration')
            for X in data_loader:
                optimizer.zero_grad()
                loss = self.model.loss(X)
                loss.backward()
                optimizer.step()
                if global_step % self.gradient_accumulation_step == 0 and global_step > 0:
                    self.summary_writer.add_scalar('Train loss', loss, global_step)
                    print('epochs:{}/{} \t global_step:{} loss:{}'.format(ep, self.epochs, global_step, loss))
                    grid = torchvision.utils.make_grid(X[:10], nrow=5)
                    self.summary_writer.add_images('images', X[:10])

                if global_step % 100 == 0 and global_step > 0:
                    # self.save_model()
                    self.sample(global_step)

                global_step += 1

        # 保存最后的模型

        self.summary_writer.add_graph(self.model.eps_model, (torch.randn(10, self.image_channels, self.image_size, self.image_size).to(self.device),
                                                             torch.randint(0, 100, (10,)).to(self.device)))
        self.save_model()
        self.summary_writer.flush()

    def save_model(self):
        if not os.path.exists(os.path.join(self.model_dir)):
            os.makedirs(self.model_dir)
        torch.save(self.model.eps_model.state_dict(), os.path.join(self.model_dir, 'ddpm.param'))
        self.logger.info('model saved success ~~~~:{}'.format(self.model_dir))
        print('model saved success ~~~~:{}'.format(self.model_dir))

    @staticmethod
    def save_pic(grid, name):
        grid = grid.permute(1, 2, 0)
        save_image(grid, name, format='CHW')

    def save_pic2(self, imgs, name):
        r = c = int(self.n_samples**.5)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(imgs[cnt, :, :, 0].cpu(), cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
                if cnt >= len(imgs):
                    break
        plt.tight_layout()
        plt.savefig(name)
        plt.close()

    def sample(self, global_step):
        if not os.path.exists('samples'):
            os.mkdir('samples')
        with torch.no_grad():
            x = torch.randn([self.n_samples, self.image_channels,
                             self.image_size, self.image_size], device=self.device)

            for t_ in range(self.n_steps):
                t = self.n_steps - t_ -1
                t = x.new_full((self.n_samples,), t).to(torch.int64)
                x = self.model.p_sample(x, t)
            # grid = torchvision.utils.make_grid(x, nrow=4)
            # self.save_pic(grid, 'samples/sample_grid_{}.png'.format(global_step))
            self.summary_writer.add_images('images_val', x, global_step, dataformats='NCHW')

            gen_imgs = x.permute(0, 2, 3, 1)
            self.save_pic2(gen_imgs, 'samples/sample_{}'.format(global_step))








