import torch
from torch import nn


class AttentionCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_length):
        super(AttentionCell, self).__init__()

        self.max_output_seq_length = output_seq_length

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.linear_h_a = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.linear_s_a = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_a = nn.Linear(self.hidden_dim, 1, bias=False)
        self.linear_init_s = nn.Linear(self.input_dim, self.hidden_dim, bias=True)
        self.linear_r = nn.Linear(self.output_dim + self.hidden_dim + self.input_dim, self.hidden_dim)
        self.linear_z = nn.Linear(self.output_dim + self.hidden_dim + self.input_dim, self.hidden_dim)
        self.linear_sp = nn.Linear(self.output_dim + self.hidden_dim + self.input_dim, self.hidden_dim)
        self.linear_y = nn.Linear(self.output_dim + self.hidden_dim + self.input_dim, self.output_dim)

    def forward(self, h):
        """
        :param h: feature map, tensor of shape (batch, num_steps, input_dim)
        :return:
        """
        self.batch, self.steps, self.input_dim = h.shape
        self.h = h

        # linear_h_a(h) does not depend upon decoding time-step i. We then compute it once and store it in wh.
        self.wh = self.linear_h_a(h)

        si = self.initHidden(h)
        yi = self.initSeq(h)

        outputs = torch.zeros([self.batch, self.max_output_seq_length, self.output_dim])
        for i in range(self.max_output_seq_length):
            si, yi = self.step(si, yi)
            outputs[:, i, :] = yi

        return outputs

    def step(self, prev_hidden, prev_output):
        """

        :param prev_hidden: hidden vector of mast step
        :param prev_output: output vector of last step
        :def
        im:      i-1 step
        s_i: cell state at stap i
        y_i: probability distribution of the ith char
        :return:
        """
        s_im = prev_hidden
        y_im = prev_output

        s_im_dup = s_im.unsqueeze(1).repeat(1, self.steps, 1)
        # duplicate s_im along input sequence dimension, s_im being dependant upon decoding time step
        a_i = self.wh + self.linear_s_a(s_im_dup)
        a_i = self.tanh(a_i)
        a_i = self.linear_a(a_i)[:, :, 0]
        a_i = self.softmax(a_i)

        c_i = a_i * self.h.permute(2, 0, 1)
        c_i = c_i.permute(1, 2, 0)
        c_i = torch.sum(c_i, 1)

        ysc = torch.cat((y_im, s_im, c_i), 1)
        r_i = self.sigmoid(self.linear_r(ysc))
        z_i = self.sigmoid(self.linear_z(ysc))
        yrsc = torch.cat((y_im, r_i * s_im, c_i), 1)
        sp_i = self.tanh(self.linear_sp(yrsc))

        s_i = (1 - z_i) * s_im + z_i * sp_i
        y_i = self.softmax(self.linear_y(ysc))

        return s_i, y_i

    def initHidden(self, x):
        s_0 = self.tanh(self.linear_init_s(x[:, 0, :]))
        return s_0

    def initSeq(self, x):
        y_0 = torch.zeros(x.shape[0], self.output_dim)
        return y_0


if __name__ == "__main__":

    batch, step, input_dim = 2, 3, 4
    hidden_attention_dim, output_dim, output_seq_length = 5, 6, 7
    inputs = torch.rand(batch, step, input_dim)

    attCell = AttentionCell(input_dim, hidden_attention_dim, output_dim, output_seq_length)
    output = attCell(inputs)

    print(output)
