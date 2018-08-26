import torch
from torch import nn
import numpy as np
import csv

from utils.CER import CER
from data.vars import Vars
from models.ANN import Net0
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
    label_error = [CER(l, p) for l, p in zip(labels_decoded, outputs_decoded)]
    label_error_mean = np.sum(label_error) / len(label_error)
    word_error = [0 if cer == 0 else 1 for cer in label_error]
    word_error_mean = np.mean(word_error)

    return loss, label_error_mean, word_error_mean


def test_model(net, name):
    print(' ')
    print("_____testing______")

    data_test = Data(V.images_test_dir, V.labels_test_txt)
    gen_test = data_test.generator(2000)
    inputs, labels = adapt_data_format(*gen_test.__next__())
    outputs = net(inputs)

    loss, label_error, word_error = errors(outputs, labels, data_test)

    with open("/%s/pytorch/%s/Test/loss.csv" % (V.experiments_folder, name), 'w') as f:
        fieldnames = ['name', 'value']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow({'name': 'cross-entropie', 'value': loss})
        print('cross-entropie : %f' % loss)
        writer.writerow({'name': 'label error', 'value': label_error})
        print('label error : %f' % label_error)
        writer.writerow({'name': 'word error', 'value': word_error})
        print('word error : %f' % word_error)

    outputs_decoded, labels_decoded = decode(outputs, labels, data_test)
    with open("%s/pytorch/%s/Test/predictions.csv" % (V.experiments_folder, name), 'w') as f:
        fieldnames = ['label', 'prediction', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for l, p in zip(labels_decoded, outputs_decoded):
            writer.writerow({'label': l, 'prediction': p, 'error': CER(l, p)})


if __name__ == "__main__":
    name = '2018-08-22-13-42-49'
    #net = Net0()
    net = torch.load("/%s/pytorch/%s/net" % (V.experiments_folder, name))
    test_model(net, name)