import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        # input of shape (seq_len, batch, input_size)
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class crnn(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        CRNN model

        :param in_channels: input pic channel
        :param out_channels: output class number
        """
        super().__init__()
        self.in_channels = in_channels
        self.nClass = out_channels # class number
        self.hidden_size = 256 # hidden units in LSTM

        # conv layers structure
        self.conv_params = ((3, 1, 1), (3, 1, 1),
                          (3, 1, 1), (3, 1, 1), (3, 1, 1),
                            (3, 1, 1), (2, 1, 0))
        # pooling layer structure
        self.pool_params = ((2, 2), (2, 2), None,
                            [(2, 2), (2, 1), (0, 1)], None,
                            [(2, 2), (2, 1), (0, 1)], None)
        # net_structure
        self.net_params = (64, 128, 256, 256, 512, 512, 512)
        # init cnn model
        self.cnn = nn.Sequential()
        # build the model
        self.build()

    def convRelu(self, idx, bn=False):
        """

        :param idx: index of layers
        :param bn: if use batchnorm

        """
        nIn = self.in_channels if idx == 0 else \
                                    self.net_params[idx - 1]
        nOut = self.net_params[idx]
        self.cnn.add_module('conv{0}'.format(idx),
                            nn.Conv2d(nIn, nOut, *(self.conv_params[idx])))
        if bn:
            self.cnn.add_module('batchnorm{0}'.format(idx),
                                nn.BatchNorm2d(nOut))

        self.cnn.add_module('relu{0}'.format(idx), nn.ReLU(True))


    def pooling(self, idx):
        if self.pool_params[idx]:
            self.cnn.add_module('pooling{0}'.format(idx),
                                nn.MaxPool2d(*self.pool_params[idx]))

    def build(self):
        self.convRelu(0)
        self.pooling(0) # 64*16*50

        self.convRelu(1)
        self.pooling(1) # 128*8*25

        self.convRelu(2)

        self.convRelu(3)
        self.pooling(3)

        self.convRelu(4, True)

        self.convRelu(5, True)
        self.pooling(5)

        self.convRelu(6)

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, self.hidden_size, self.hidden_size),
            BidirectionalLSTM(self.hidden_size, self.hidden_size, self.nClass)
        )

    def forward(self, input): # input: height=32, width>=100
        conv_out = self.cnn(input) # batch, channel=512, height=1, widths>=24
        b, c, h, w = conv_out.size()

        assert h == 1, "the output height of cnn must be 1, got shape:b:{} c:{} h:{} w:{}".format(b, c, h, w)

        # feature sequence extraction from feature maps
        # each feature vector of a feature sequence is generated from left to right on the feature maps by column
        conv_out = conv_out.squeeze(2) # [batch, channel, width]
        # permute dims in order to fit nn.LSTM
        conv_out = conv_out.permute(2, 0, 1) # [width, batch, channel]

        output = self.rnn(conv_out)
        return output

