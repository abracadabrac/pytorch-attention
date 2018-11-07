from models_1.Decoder import Decoder
from models_1.Encoder import Encoder
from models_1.Attention import Attention

import torch
from torch import nn


class EncoderDecoder(nn.Module):
    def __init__(self, input_dim, feature_map_dimension, dec_state_size, output_vocab_size,
                 output_sec_length=25,
                 attention_method='gen'):
        super(EncoderDecoder, self).__init__()
        self.output_sec_length = output_sec_length

        self.enc = Encoder(input_dim, feature_map_dimension)

        self.attention = Attention(feature_map_dimension, dec_state_size, attention_method)

        self.dec = Decoder(feature_map_dimension, dec_state_size, output_vocab_size)

        self.ftMapEmbedding = nn.Linear(feature_map_dimension, dec_state_size)
        self.stateEmbedding = nn.Linear(dec_state_size, dec_state_size)

    def forward(self, input_image):
        batch_size, _, input_sec_length, input_dimension = input_image.shape

        enc_hidden = self.enc.init_hidden(batch_size)
        feature_map, enc_hidden = self.enc(input_image, enc_hidden)

        embedded_feature_map = self.ftMapEmbedding(feature_map)

        dec_state, dec_pred = self.dec.init_state(feature_map), self.dec.init_pred(batch_size)
        output = torch.zeros(batch_size, self.output_sec_length, self.dec.output_vocab_size)
        for i in range(self.output_sec_length):

            embedded_state = self.stateEmbedding(dec_state)
            attention_map = self.attention(embedded_feature_map, embedded_state)

            dec_pred, dec_state = self.dec(feature_map, dec_pred, dec_state, attention_map)
            output[:, i, :] = dec_pred

        return output