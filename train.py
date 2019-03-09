import torch
import os
import argparse
import torch.optim as optim
from warpctc_pytorch import CTCLoss
from model.crnn import crnn
from dataset import load_data
from utils import *
from torchsummary import summary

parser = argparse.ArgumentParser()
parser.add_argument('--root', type=str, default='/workspace/xzj/datasets/IIIT5K', help='path to the dataset')
parser.add_argument('--batch_size', type=int, default=128, help = 'input batch size')
parser.add_argument('--se', type=int, default=0, help='starting epoch')
parser.add_argument('--epoch', type=int, default=20, help='number of epoch to train(default=20)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--saveInterval', type=int, default=5, help='Interval the model to be saved')
parser.add_argument('--pretrained',default='', help='path to pretrained model(continuing training)')
parser.add_argument('--exp_dir', default='exp', help='folder to store samples and model')
parser.add_argument('--alphabet', type=str, default='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
opt = parser.parse_args()
print(opt)

# dir to save model
if not os.path.exists(opt.exp_dir):
    os.mkdir(opt.exp_dir)

# cuda devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# custom weight initialize
def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('batchnorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def train(epoch_num, start_epoch=0, preTrained=opt.pretrained,
          lr=0.1, print_every=8):

    train_loader = load_data(opt.root, batch_size=opt.batch_size)
    print(len(train_loader))
    print('successfully load data')
    assert train_loader

    net = crnn(1, len(opt.alphabet) + 1)  # in_channel = 1, out_channel = nClass
    net.apply(weight_init)

    if preTrained != '':
        print('loading Pre_trained model from %s' % preTrained)
        net.load_state_dict(torch.load(preTrained))
        print('Successfully load Pre_trained Model')
    # print(net)

    # loss func
    criterion = CTCLoss()
    # optimizer
    optimizer = optim.Adadelta(net.parameters(), lr=lr, weight_decay=1e-3)

    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    if use_cuda:
        print('GPU detected')
        net = net.to(device)
        criterion = criterion.to(device)
    else:
        print('*** cuda is not available, switch to cpu ***')
    # summary(net, input_size=(1, 32, 100))

    label_encoder = str2LabelConverter(opt.alphabet) # initialize encoder and decoder

    print('===Start training...===')
    net.train() #
    for e in range(start_epoch, epoch_num):
        print('-------  epoch: %d  ------'% (e + 1))
        for idx, (img, label) in enumerate(train_loader):
            labelEncoded, label_length = label_encoder.encode(label)
            img = img.to(device)
            if e == 0 and idx == 0:
                print('input shape = {0}, {1}, {2}, {3}'.format(*(img.size())) )

            optimizer.zero_grad()
            pre = net(img) # batch of predicted texts
            pre_length = torch.IntTensor(
                [pre.size(0)] * pre.size(1)
            ) # pre_length [n]
            loss = criterion(pre, labelEncoded, pre_length, label_length)
            # update
            loss.backward()
            optimizer.step()

            if idx % print_every == 0:
                print('Iteration %d, loss = %.4f' % (idx, loss))

        if e % opt.saveInterval == 0:
            torch.save(
                net.state_dict(), '{0}/CRNN_{1}.pth'.format(opt.exp_dir, e)
            )

    print('Finished Training!')
    torch.save(net.state_dict(),'{0}/CRNN_Final'.format(opt.exp_dir))

if __name__ == '__main__':
    train(opt.epoch, start_epoch=opt.se, lr=opt.lr)
