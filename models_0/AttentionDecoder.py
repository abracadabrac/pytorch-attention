import torch
from torch import nn

"""
What follows in an implementation of an attention cell.
It reproduces the algorithm presented by Dzmitry Bahdanau, KyungHyun Cho and Yoshua Bengio in NEURAL MACHINE TRANSLATION
BY JOINTLY LEARNING TO ALIGN AND TRANSLATE.



      '||                    '||`              
       ||                     ||               
.|'',  ||''|,  '''|.  '||''|  ||  .|''|, ('''' 
||     ||  || .|''||   ||     ||  ||..||  `'') 
`|..' .||  || `|..||. .||.   .||. `|...  `...' 
                                               


"""


class AttentionDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, output_seq_length):
        super(AttentionDecoder, self).__init__()

        self.max_output_seq_length = output_seq_length

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

        self.linear_h_a = nn.Linear(input_dim, self.hidden_dim, bias=False)
        self.linear_s_a = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.linear_a = nn.Linear(2 * self.hidden_dim, 1, bias=False)
        self.linear_init_s = nn.Linear(self.input_dim, self.hidden_dim, bias=True)

        self.gru0 = nn.GRUCell(self.output_dim + self.input_dim, self.hidden_dim)
        self.rnn0 = nn.RNNCell(self.hidden_dim + self.input_dim, self.output_dim)

    def forward(self, h):
        """
        :param h: feature map, tensor of shape (batch, num_steps, input_dim)
        :return:
        """
        self.batch, self.input_steps, self.input_dim = h.shape
        self.h, self.wh = h, self.linear_h_a(h)
        # linear_h_a(h) does not depend upon decoding time-step i. We then compute it once and store it in wh.
        si, yi = self.initHidden(h), self.initSeq()

        outputs = torch.zeros([self.batch, self.max_output_seq_length, self.output_dim])
        for i in range(self.max_output_seq_length):
            si, yi = self.step(si, yi)
            outputs[:, i, :] = yi

        return outputs

    def step(self, sim, yim):
        """
        :param prev_hidden: hidden vector of mast step
        :param prev_output: output vector of last step
        :def
        im:  i-1th step in the output sequence
        s_i: cell state                                     (hidden_dim)
        y_i: probability distribution of the ith char       (output_dim) = (vocab_size)
            -> should be noted p(y_i) or py_i
        c_i: context vector                                 (input_dim)
        a_i: attention map                                  (input_seq_length)
        :return:
        """

        sim_dup = sim.unsqueeze(1).repeat(1, self.input_steps, 1)
        # duplicate s_im along input sequence dimension, s_im being dependant upon decoding time step
        ai = self.tanh(torch.cat((self.wh, self.linear_s_a(sim_dup)), 2))
        ai = self.softmax(self.linear_a(ai)[:, :, 0])

        ci = ai.unsqueeze(1).bmm(self.h).squeeze()

        input_gru = torch.cat((ci, yim), 1)
        hidden_gru = sim
        si = self.gru0(input_gru, hidden_gru)

        input_rnn = torch.cat((ci, si), 1)
        hidden_rnn = yim
        yi = self.softmax(self.rnn0(input_rnn, hidden_rnn))

        return si, yi

    def initHidden(self, x):
        s_0 = self.tanh(self.linear_init_s(x[:, 0, :]))
        return s_0

    def initSeq(self):
        y_0 = torch.zeros(self.batch, self.output_dim)
        return y_0


if __name__ == "__main__":

    batch, step, input_dim = 2, 3, 4
    hidden_attention_dim, output_dim, output_seq_length = 5, 6, 7
    inputs = torch.rand(batch, step, input_dim)

    attCell = AttentionDecoder(input_dim, hidden_attention_dim, output_dim, output_seq_length)
    output = attCell(inputs)

    print(output)
