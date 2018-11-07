from torch import nn
import torch
import torch.functional as F


class Attention(nn.Module):
    def __init__(self, input_size, state_size, attention_method):
        super(Attention, self).__init__()

        self.attention_method = attention_method
        self.input_size = input_size

        if self.attention_method == 'gen':
            self.tanh = nn.Tanh()
            self.l1 = nn.Linear(state_size, 1)

        self.softmax = nn.Softmax(1)

    def forward(self, input_sec, state):
        batch = input_sec.shape[0]
        sec_length = input_sec.shape[1]
        attention_map = torch.zeros(batch, sec_length)
        for i in range(sec_length):
            attention_map[:, i] = self._score(state, input_sec[:, i, :])

        return self.softmax(attention_map)

    def _score(self, state, x):
        if self.attention_method == 'dot':
            energy = state.unsqueeze(1).bmm(x.unsqueeze(2))

        if self.attention_method == 'gen':
            energy = self.l1(self.tanh(x + state))

        return energy.squeeze()
