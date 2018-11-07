import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable


class Decoder(nn.Module):
    def __init__(self, input_size, state_size, output_vocab_size):
        """

        :param input_size: dimension of the feature map
        :param attention_size:
        :param state_size: dimension of the state vector s_i
        :param attention_method: alignment can be obtained with different score function 'gen' or 'dot'
        cf : "Effective Approaches to Attention-based Neural Machine Translation" by Minh-Thang Luong et al.
        """
        super(Decoder, self).__init__()

        self.input_size = input_size
        self.state_size = state_size
        self.output_vocab_size = output_vocab_size

        self.stateUpdate = nn.GRUCell(input_size + output_vocab_size, state_size)

        self.outputProducer = nn.RNNCell(state_size + input_size, output_vocab_size)
        self.softmax = nn.Softmax(1)

        self.initState = nn.Linear(input_size, state_size)

    def forward(self, input, prev_pred, prev_state, attention_map):
        """
        :param input: feature map
        :param prev_pred: can be either the probability distribution over the vocabulary given by the decoder at the
        last step p(y_im) or the encoding of the prediction encoding[argmax[p(y_im)]]
        :param prev_context: context vector computed at the last time step
        :param prev_state: state vector at the last time step
        :return:
        """

        # bbm: batch matrix multiplication
        context = attention_map.unsqueeze(1).bmm(input).squeeze(1)

        state = self.stateUpdate(torch.cat((context, prev_pred), 1), prev_state)

        prob_distr = self.outputProducer(torch.cat((context, state), 1), prev_pred)
        prob_distr = self.softmax(prob_distr)

        return prob_distr, state

    def init_pred(self, batch_size):
        init_pred = Variable(torch.zeros(batch_size, self.output_vocab_size), requires_grad=True)
        return init_pred

    def init_state(self, input):
        init_state = F.tanh(self.initState(input[:, 0, :]))
        return init_state


if __name__ == "__main__":
    from models_1.Encoder import Encoder

    batch = 3
    image_length = 384
    image_high = 28
    input = torch.randn([batch, 1, image_length, image_high])

    feature_map_dimension = 200
    enc = Encoder(feature_map_dimension)
    init_hidden = enc.init_hidden(batch)

    output_enc, hidden_enc = enc(input, init_hidden)

    state_size_decoder = feature_map_dimension
    output_vocab_size = 30
    dec = Decoder(feature_map_dimension, state_size_decoder, output_vocab_size, attention_method='gen')

    input_dec = output_enc
    init_pred = dec.init_pred()
    init_state = hidden_enc.squeeze(0)

    dec(input_dec, init_pred, init_state)

    print("hello charles,")