import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


def init_conv_weights(m):
    if type(m) == nn.Conv2d:
        m.weight.data = torch.randn(m.weight.data.shape) * 0.2

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dimension):
        """
        Implement a
        :param hidden_size:
        :param input_size:
        :param sec_length:
        """
        super(Encoder, self).__init__()

        self.hidden_size = int(feature_dimension / 2)

        self.conv0 = nn.Conv2d(1, 8, 3, padding=1)
        self.conv1 = nn.Conv2d(8, 16, 3, padding=1)
        self.maxp0 = nn.MaxPool2d((3, 2))
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.maxp1 = nn.MaxPool2d((4, 2))
        self.apply(init_conv_weights)

        self.hidden_shape = [int(input_dim[0] / (4 * 3)),
                             int(input_dim[1] / (2 * 2) * 32)]

        self.gru = nn.GRU(self.hidden_shape[1], self.hidden_size, 1, batch_first=True, bidirectional=True)
        # feature dimension divised by two because the GRU is bidirectional
    def forward(self, input_image, hidden):
        """
        :param input_image: in dim [batch_size, 1 (the channel dim), sec_length, input_size]
        :param hidden: produced by self.init_hidden
        :return: output in dim [batch_size, sec_length, hidden_dim]
        """
        batch_size = len(input_image)
        y0 = F.relu(self.conv0(input_image))
        y1 = F.relu(self.conv1(y0))

        y2 = self.maxp0(y1)

        y3 = F.relu(self.conv2(y2))
        y4 = F.relu(self.conv3(y3))

        y5 = self.maxp1(y4)

        gru_input_sec = self.reshape(y5)

        # another reshape solution
        # y3 = y2.permute(0, 2, 1, 3).contiguous().view(batch_size, self.hidden_shape[0], self.hidden_shape[1])

        output, hidden = self.gru(gru_input_sec, hidden)

        return output, hidden

    def reshape(self, y_in):
        # This (ugly) function implement a reshape operation. This way i am sure that the right operation is done.
        batch_size = len(y_in)
        y_out = torch.zeros([batch_size, self.hidden_shape[0], self.hidden_shape[1]])
        vertical_position = -1
        for i in range(self.hidden_shape[1]):
            channel_num = i % 32
            if channel_num == 0: vertical_position += 1
            y_out[:, :, i] = y_in[:, channel_num, :, vertical_position]

        return y_out

    def init_hidden(self, batch_size):
        num_directions = 2      # bidirectional GRU
        hidden = Variable(torch.zeros(num_directions, batch_size, self.hidden_size), requires_grad=True)
        return hidden


if __name__ == "__main__":
    enc = Encoder([28, 384], 200)
    input = torch.randn([3, 1, 384, 28])
    hidden = enc.init_hidden(3)

    output, hidden = enc(input, hidden)

    print("hello charles,")
