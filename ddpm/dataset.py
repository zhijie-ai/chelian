import torch
from torch.utils.data import Dataset, IterableDataset
import torchvision
import torchvision.transforms as trans
import os
import cv2

class DDPMDataset(Dataset):
    def __init__(self, device, mode='cifar10'):
        super().__init__()
        self.device = device
        self.pad = torch.nn.ConstantPad2d((2, 2, 2, 2), 0)
        self.mode = mode
        self.__get_data()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        if self.mode == 'minist':
            return self.format_minist(item)
        elif self.mode == 'celeba':
            return self.format_celeba(item)
        elif self.mode == 'cifar10':
            return self.format_cifa10(item)

    def __get_data(self):
        assert self.mode in ['minist', 'celeba', 'cifar10']
        if self.mode == 'minist':
            data = torchvision.datasets.MNIST('./data', download=True, train=True)
            self.X, self.y = data.data, data.targets
        elif self.mode == 'celeba':
            self.X = self.read_celeba()
        elif self.mode == 'cifar10':
            transformer = trans.Compose([trans.ToTensor()])
            data = torchvision.datasets.CIFAR10('./data', download=True, train=True, transform=transformer)
            self.X, self.y = data.data, data.targets
            self.X = torch.tensor(self.X).permute(0,3,1,2)

    def format_minist(self, ind):
        sample = self.X[ind]
        sample = self.pad(sample)
        sample = sample.unsqueeze(0).to(torch.float32).to(self.device)
        return sample

    def format_celeba(self, ind):
        sample = self.X[ind]
        sample = torch.tensor(sample)
        return sample.to(torch.float32).to(self.device)


    def format_cifa10(self, ind):
        sample = self.X[ind]
        return sample.to(torch.float32).to(self.device)

    def read_celeba(self):
        i = 0
        X = []
        path = 'data/celeba/img_align_celeba'
        f_list = os.listdir(path)
        for f in f_list:
            if i >= 50000:
                break
            img = cv2.imread(os.path.join(path, f))
            img = cv2.resize(img, (128, 128))
            X.append(img)
            i += 1

        return X


class CelebaDataset(IterableDataset):
    def __init__(self,path):
        self.path = path

    def __iter__(self):
        return self.read_img()

    def read_img(self):
        path = 'data/celeba/img_align_celeba'
        f_list = os.listdir(self.path)
        for f in f_list:
            img = cv2.imread(os.path.join(path, f))
            img = cv2.resize(img, (128, 128))
            yield img

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = DDPMDataset('cpu', mode='celeba')
    print(dataset[0].shape, type(dataset[0]))
    cv2.imwrite('a.png', dataset[1].numpy())
    plt.imshow(dataset[1])
    plt.show()



