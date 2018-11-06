import torch
from torch import nn, optim
import numpy as np
import datetime
import os
import pandas as pd

from models.ANN import EncoderDecoder
from
from data.reader import Data
from data.vars import Vars
from Test_model import test_model, errors, save_in_files
from utils.run_utils import adapt_data_format, save_loss


V = Vars()


def train_model_1(enc, dec, name, nb_epoch=1, nb_batch=1, batch_size=1, lr=1e-4):
    print(' ')
    print("_____training_____")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    data_train = Data(V.images_train_dir, V.labels_train_txt)
    data_valid = Data(V.images_valid_dir, V.labels_valid_txt)
    best_loss_valid = 666
    learning_data = []
    loss_train_cumulate = []

    for epoch in range(1, nb_epoch):
        gen_train = data_train.generator(batch_size)
        print(' ')
        print('epoch : %d' % epoch)

        inputs, labels = gen_train.__next__()
        batch = 1
        while inputs.shape[0] != 0:
            inputs, labels = adapt_data_format(inputs, labels)      # the Data class was first developed for a keras
            #  model and needs adjustment to be used with the torch one
            optimizer.zero_grad()

            outputs = net(inputs)

            loss = nn.functional.cross_entropy(outputs, labels)
            loss_train_cumulate.append(loss.item())
            loss.backward()
            optimizer.step()

            if batch % 100 == 99:
                print(' ')
                print(' epoch  %d/%d,    batch  %d/%d' % (epoch, nb_epoch, batch, nb_batch))

                gen_valid = data_valid.generator(300)
                inputs, labels = adapt_data_format(*gen_valid.__next__())

                outputs = net(inputs)

                loss_train = np.mean(loss_train_cumulate)
                loss_train_cumulate = []
                loss_valid, label_error, word_error = errors(outputs, labels, data_valid)
                learning_data.append({'epoch': epoch,
                                      'batch': batch,
                                      'loss_valid': loss_valid,
                                      'loss_train': loss_train,
                                      'label_error': label_error,
                                      'word_error': word_error})
                learing_summary = pd.DataFrame(learning_data)
                learing_summary.to_csv(path_or_buf='%s/pytorch/%s/learning_summary.csv' % (V.experiments_folder, name))

                path = "%s/pytorch/%s/Training/e%d-b%d/" % (V.experiments_folder, name, epoch, batch)
                os.makedirs(path)
                save_in_files(path, loss_valid, label_error, word_error, outputs, labels, data_valid)

                print('     loss_valid    %f' % loss_valid)
                print('     loss_train    %f' % loss_train)
                print('     label_error   %f' % label_error)
                print('     word_error    %f' % word_error)

                if loss_valid < best_loss_valid:
                    print('     *  new best loss valid    %f' % loss_valid)
                    torch.save(net, '%s/pytorch/%s/Weights/net-e%d-b%d' % (V.experiments_folder, name, epoch, batch))
                    best_loss_valid = loss_valid

            inputs, labels = gen_train.__next__()
            batch += 1

    return net


if __name__ == "__main__":
    now = datetime.datetime.now().replace(microsecond=0)
    name = datetime.date.today().isoformat() + '-' + now.strftime("%H-%M-%S")

    os.makedirs("/%s/pytorch/%s/Test" % (V.experiments_folder, name))
    os.makedirs("/%s/pytorch/%s/Weights" % (V.experiments_folder, name))
    nb_epoch = 10
    batch_size = 8
    nb_batch = int(len(os.listdir(V.images_train_dir)) / batch_size)
    lr = 1e-3

    enc = Encoder()
    dec = Decoder()
    net = train_model_1(enc, dec, name, nb_epoch=nb_epoch, nb_batch=nb_batch, batch_size=batch_size, lr=lr)

    test_model(net, name)


