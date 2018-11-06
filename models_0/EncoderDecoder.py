import torch
from torch import nn
import torch.nn.functional as F
from models_0.AttentionDecoder import AttentionDecoder

import numpy as np

torch.manual_seed(1.2)


"""
 
      '||                    '||`              
       ||                     ||               
.|'',  ||''|,  '''|.  '||''|  ||  .|''|, ('''' 
||     ||  || .|''||   ||     ||  ||..||  `'') 
`|..' .||  || `|..||. .||.   .||. `|...  `...' 
                                               
"""


def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.data = torch.randn(m.weight.data.shape) * 0.2


class EncoderDecoder(nn.Module):
    def __init__(self):
        super(EncoderDecoder, self).__init__()

        self.input_shape = [384, 28]
        self.strides = [(2, 1), (2, 2), (3, 2), (1, 1)]
        self.strides_total = np.product(self.strides, axis=0)
        self.cc = [4, 16, 16, 32]          # convolution channels

        self.length_sec = int(self.input_shape[0] / self.strides_total[0])
        self.height_image = int(self.input_shape[1] / self.strides_total[1])
        self.abstract_dim = int(self.height_image * self.cc[3])

        self.conv1 = nn.Conv2d(1, self.cc[0], 3, padding=1)
        self.maxp1 = nn.MaxPool2d(self.strides[0])
        self.conv2 = nn.Conv2d(self.cc[0], self.cc[1], 3, padding=1)
        self.maxp2 = nn.MaxPool2d(self.strides[1])
        self.conv3 = nn.Conv2d(self.cc[1], self.cc[2], 3, padding=1)
        self.maxp3 = nn.MaxPool2d(self.strides[2])
        self.conv4 = nn.Conv2d(self.cc[2], self.cc[3], 3, padding=1)
        self.maxp4 = nn.MaxPool2d(self.strides[3])
        self.apply(init_conv_weights)

        hidden_lstm_size = self.abstract_dim
        input_lstm_size = self.abstract_dim
        self.hidden_0 = (torch.randn(1, hidden_lstm_size, requires_grad=True),
                         torch.randn(1, hidden_lstm_size, requires_grad=True))
        self.lstm0 = nn.LSTM(input_lstm_size, hidden_lstm_size, bidirectional=False)

        hidden_attention_size = 256
        input_attention = self.abstract_dim
        self.attention = AttentionDecoder(input_attention, hidden_attention_size, 29, 25)

    def forward(self, x):
        batch_size = x.shape[0]
        y1 = self.maxp1(F.relu(self.conv1(x)))
        y2 = self.maxp2(F.relu(self.conv2(y1)))
        y3 = self.maxp3(F.relu(self.conv3(y2)))
        y4 = self.maxp4(F.relu(self.conv4(y3)))

        y5 = torch.zeros([batch_size, self.length_sec, self.abstract_dim])
        vertical_position = -1
        for i in range(self.abstract_dim):
            channel_num = i % self.cc[-1]
            if channel_num == 0:
                vertical_position += 1

            y5[:, :, i] = y4[:, channel_num, :, vertical_position]

        hidden = (self.hidden_0[0].unsqueeze(1).repeat(1, batch_size, 1),
                  self.hidden_0[1].unsqueeze(1).repeat(1, batch_size, 1))

        y7, hidden = self.lstm0(y5.permute(1, 0, 2), hidden)
        y8 = y7.permute(1, 0, 2)

        y9 = self.attention(y8).permute(0, 2, 1)

        return y9


if __name__ == '__main__':
    net = EncoderDecoder()

    inputs = torch.rand(5, 1, 384, 28)
    o = net.forward(inputs)
