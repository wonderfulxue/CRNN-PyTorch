import os
import torch
from torch.utils.data import Dataset
from torch .utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import scipy.io as sio
import pickle
import numpy as np

class ResizeImg(object):
    """
    scale image to fixed height
    """

    def __init__(self, height=32, minWidth=100):
        self.height = height
        self.minWidth = minWidth

    # img is PIL.image instance
    def __call__(self, img):
        w, h = img.size
        width = int(np.max(int(w * self.height / h), self.minWidth))
        return img.resize((width, self.height), Image.ANTIALIAS)


class IIIT5k(Dataset):
    """
    IIIT-5K dataset
    """

    def __init__(self, root_dir=None, training=True):
        super().__init__()
        data_str = 'traindata' if training else 'testdata'
        self.root_dir = root_dir
        # data 由 4项组成，第1项：图片文件名 第2项：label 第3项：大小=50的字典 第4项：大小=1000的字典
        self.data = sio.loadmat(os.path.join(root_dir, data_str + '.mat'))[data_str][0]
        # resize image + gray scale + transform to tensor
        self.transform = [transforms.Resize((32, 100), Image.ANTIALIAS)]
        self.transform.extend([transforms.Grayscale(), transforms.ToTensor()])
        self.transform = transforms.Compose(self.transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx][0][0]
        img_label = self.data[idx][1][0]

        img = self.transform(Image.open(self.root_dir + '/' + img_name))

        return img, img_label


def load_data(root_dir=None, training=True, batch_size=64):
    """
    load IIIT-5K dataset
    :param root_dir: root dir of the dataset
    :param training: if True, train the model, else, vice versa
    :param batch_size: batch size, default is 128 img per batch
    :return: Training set or test set based on the training flag
    """
    if training:
        filename = os.path.join(root_dir, 'train' + '.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            print('==== Loading data... ===')
            dataset = IIIT5k(root_dir)
            pickle.dump(dataset, open(filename, 'wb'))
        dataLoader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=True, num_workers=4)

    else:
        filename = os.path.join(root_dir, 'test' + '.pkl')
        if os.path.exists(filename):
            dataset = pickle.load(open(filename, 'rb'))
        else:
            dataset = IIIT5k(root_dir, training=False)
            pickle.dump(dataset, open(filename, 'wb'))
        dataLoader = DataLoader(dataset, batch_size=batch_size
                                ,shuffle=False)
    return dataLoader






