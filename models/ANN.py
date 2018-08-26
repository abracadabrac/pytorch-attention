import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention_cell import AttentionCell

import numpy as np

torch.manual_seed(1.2)


def repeat_hidden_along_batch(hidden, batch_size):
    return (hidden[0].unsqueeze(1).repeat(1, batch_size, 1),
            hidden[1].unsqueeze(1).repeat(1, batch_size, 1))


def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.data = torch.randn(m.weight.data.shape) * 0.1


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.input_shape = [384, 28]
        self.strides = [(2, 1), (2, 2), (3, 2), (1, 1)]
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.maxp1 = nn.MaxPool2d(self.strides[0])
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1)
        self.maxp2 = nn.MaxPool2d(self.strides[1])
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.maxp3 = nn.MaxPool2d(self.strides[2])
        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxp4 = nn.MaxPool2d(self.strides[3])
        self.apply(init_conv_weights)
        self.hidden_0 = (torch.randn(1, 128, requires_grad=True),
                         torch.randn(1, 128, requires_grad=True))
        self.lstm0 = nn.LSTM(224, 128, bidirectional=False)
        self.attention = AttentionCell(128, 128, 29, 25)

    def forward(self, x):
        batch_size = x.shape[0]
        y1 = self.maxp1(F.relu(self.conv1(x)))
        y2 = self.maxp2(F.relu(self.conv2(y1)))
        y3 = self.maxp3(F.relu(self.conv3(y2)))
        y4 = self.maxp4(F.relu(self.conv4(y3)))

        strides_total = np.product(self.strides, axis=0)
        y5 = y4.view(batch_size,
                     int(self.input_shape[0] / strides_total[0]),
                     int(self.input_shape[1] / strides_total[1] * 32))
        y6 = y5.permute(1, 0, 2)        # (seq_length, batch, encoding_dim)

        hidden = repeat_hidden_along_batch(self.hidden_0, batch_size)
        y7, hidden = self.lstm0(y6, hidden)

        y8 = y7.permute(1, 0, 2)
        y9 = self.attention(y8).permute(0, 2, 1)

        return y9


if __name__ == '__main__':
    net = Net0()

    inputs = torch.rand(5, 1, 384, 28)
    o = net.forward(inputs)

    print(o.shape)
