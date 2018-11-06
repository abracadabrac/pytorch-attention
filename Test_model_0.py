import torch
from torch import nn
import numpy as np
import csv
import os

from utils.CER import CER
from data.vars import Vars
from data.reader import Data
from utils.run_utils import adapt_data_format

V = Vars()


def decode(outputs, labels, data_test):
    outputs_decoded = torch.argmax(outputs, dim=1)
    outputs_decoded = data_test.decode_labels(outputs_decoded, depad=True, deep_lib='pytorch')
    labels_decoded = data_test.decode_labels(labels, depad=True, deep_lib='pytorch')

    return outputs_decoded, labels_decoded


def errors(outputs, labels, data_test):
    outputs_decoded, labels_decoded = decode(outputs, labels, data_test)

    loss = nn.functional.cross_entropy(outputs, labels).item()
    label_error = [CER(l, p) / len(l) for l, p in zip(labels_decoded, outputs_decoded)]
    label_error_mean = np.mean(label_error)
    word_error = [0 if cer == 0 else 1 for cer in label_error]
    word_error_mean = np.mean(word_error)

    return loss, label_error_mean, word_error_mean


def save_in_files(path, loss, label_error, word_error, outputs, labels, data_test):
    with open(path + "loss.csv", 'w') as f:
        fieldnames = ['name', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'name': 'cross-entropie', 'value': loss})
        writer.writerow({'name': 'label error', 'value': label_error})
        writer.writerow({'name': 'word error', 'value': word_error})

    outputs_decoded, labels_decoded = decode(outputs, labels, data_test)
    with open(path + "prediction.csv", 'w') as f:
        fieldnames = ['label', 'prediction', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for l, p in zip(labels_decoded, outputs_decoded):
            writer.writerow({'label': l, 'prediction': p, 'error': CER(l, p)})


def test_model(net, name):
    print(" ")
    print("_____testing %s______" % name)

    data_test = Data(V.images_test_dir, V.labels_test_txt)
    gen_test = data_test.generator(3000)
    inputs, labels = adapt_data_format(*gen_test.__next__())
    outputs = net(inputs)

    loss, label_error, word_error = errors(outputs, labels, data_test)

    path = '/%s/pytorch/%s/Test/' % (V.experiments_folder, name)
    path_loss = path + 'loss.csv'
    path_prediction = path + 'prediction.csv'
    save_in_files(path_loss, path_prediction, loss, label_error, word_error, outputs, labels, data_test)


if __name__ == "__main__":
    for name_xp in os.listdir("/%s/pytorch/" % V.experiments_folder):
        if name_xp[0] == '2':
            name_net_weigths = os.listdir("/%s/pytorch/%s/Weights" % (V.experiments_folder, name_xp))[-1]
            net = torch.load("/%s/pytorch/%s/Weights/%s" % (V.experiments_folder, name_xp, name_net_weigths))

            test_model(net, name_xp)
