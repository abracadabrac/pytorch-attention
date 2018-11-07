import torch
from torch import nn, optim
from torch.nn import functional as F

import numpy as np
import datetime
import os
import pandas as pd
import json

from data.reader import Data
from data.vars import Vars
from Test_model_1 import test_model, errors, save_in_files
from utils.run_utils import adapt_data_format

from models_1.EncoderDecoder import EncoderDecoder

V = Vars()


def train_model_1(net, name,
                  data_train, data_valid,
                  nb_epoch, nb_batch, batch_size,
                  lr, optimizer,
                  attention_method, loss_function):
    print(' ')
    print("_____training_____")

    os.makedirs("/%s/pytorch/%s/Test" % (V.experiments_folder, name))
    os.makedirs("/%s/pytorch/%s/Weights" % (V.experiments_folder, name))

    with open(V.experiments_folder + '/pytorch/' + name + '/meta_parameters.json', 'w') as f:
        json.dump({'batch_size': batch_size,
                   'learning_rate': lr,
                   'optimizer': optimizer,
                   'attention_method': attention_method,
                   'loss_function': loss_function}, f)

    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr)

    if loss_function == 'cross_entropy':
        loss_function = F.cross_entropy
    elif loss_function == 'NLLLoss':
        loss_function = nn.NLLLoss()

    output_sec_length = data_train.lb_length       # 25
    best_loss_valid = 666
    learning_data = []
    # every batch loss is saved in loss_train_cumulate. It is averaged during lavidation.
    loss_train_cumulate = []

    for epoch in range(1, nb_epoch):
        gen_train = data_train.generator(batch_size)
        print(' ')
        print('epoch : %d' % epoch)

        inputs, labels = gen_train.__next__()
        batch = 1
        while inputs.shape[0] != 0:
            # the Data class was first developed for a keras
            # model and needs adjustment to be used with the torch one
            inputs, labels, labels_argmax = adapt_data_format(inputs, labels)
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = np.sum([loss_function(outputs[:, i, :], labels_argmax[:, i]) for i in range(output_sec_length)])
            loss_train_cumulate.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch % 100 == 99:
                # enter this loop every 100 batches to do validation and saving
                print(' ')
                print(' epoch  %d/%d,    batch  %d/%d' % (epoch, nb_epoch, batch, nb_batch))

                valid_batch_size = 300

                gen_valid = data_valid.generator(valid_batch_size)
                inputs, labels, labels_argmax = adapt_data_format(*gen_valid.__next__())

                outputs = net(inputs)

                loss_valid = np.sum([loss_function(outputs[:, i, :], labels_argmax[:, i]) for i in range(output_sec_length)])
                loss_train = np.mean(loss_train_cumulate)
                loss_train_cumulate = []
                label_error, word_error = errors(outputs, labels_argmax, data_valid)
                learning_data.append({'epoch': epoch,
                                      'batch': batch,
                                      'loss_train': loss_train,
                                      'loss_valid': loss_valid.item(),
                                      'label_error': label_error,
                                      'word_error': word_error})
                learing_summary = pd.DataFrame(learning_data)
                learing_summary.to_csv(path_or_buf='%s/pytorch/%s/learning_summary.csv' % (V.experiments_folder, name))

                path = "%s/pytorch/%s/Training/e%d-b%d/" % (V.experiments_folder, name, epoch, batch)
                os.makedirs(path)
                save_in_files(path, loss_valid, label_error, word_error, outputs, labels_argmax, data_valid)

                print('     loss_valid    %f' % loss_valid)
                print('     loss_train    %f' % loss_train)
                print('     label_error   %f' % label_error)
                print('     word_error    %f' % word_error)

                # the encoder and decoder are saved every time the valid loss reaches a new minimum
                if loss_valid < best_loss_valid:
                    print('     *  new best loss valid    %f' % loss_valid)
                    torch.save(net, '%s/pytorch/%s/Weights/net-e%d-b%d' % (V.experiments_folder, name, epoch, batch))
                    best_loss_valid = loss_valid

            inputs, labels = gen_train.__next__()
            batch += 1

    return net


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # the name of the experiment folder is the date in format year-month-day-hour-minute-second
    now = datetime.datetime.now().replace(microsecond=0)
    name = datetime.date.today().isoformat() + '-' + now.strftime("%H-%M-%S")

    # experiment setup
    nb_epoch = 10
    batch_size = 8
    nb_batch = int(len(os.listdir(V.images_train_dir)) / batch_size)
    lr = 1e-3
    optimizer = 'Adam'
    attention_method = 'gen'        # 'dot'
    loss_function = 'cross_entropy'

    data_train = Data(V.images_train_dir, V.labels_train_txt)
    data_valid = Data(V.images_valid_dir, V.labels_valid_txt)

    input_dim = [data_train.im_length, data_train.im_height]
    feature_map_dim = 224
    dec_state_size = 256
    output_vocab_size = data_train.vocab_size

    net = EncoderDecoder(input_dim, feature_map_dim, dec_state_size, output_vocab_size)

    net = train_model_1(net, name,
                        data_train=data_train, data_valid=data_valid,
                        nb_epoch=nb_epoch, nb_batch=nb_batch, batch_size=batch_size,
                        lr=lr, optimizer=optimizer,
                        attention_method=attention_method, loss_function=loss_function)

    test_model(net, name)
