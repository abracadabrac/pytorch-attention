import numpy as np
import random
import os
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils.run_utils import adapt_data_format

from data.vars import Vars

V = Vars()


def pad_images(images, im_height, im_length):
    padded_images = []
    for image in images:
        padded_image = np.zeros([im_height, im_length])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                padded_image[i, j] = image[i, j]

        padded_images.append(np.rot90(padded_image, k=-1))

    return padded_images


def pad_character(char_list, length):
    char_list = [xi + '_' * (length - len(xi))
                 for xi in char_list]
    return char_list


def depad_character(char_list):
    """
    this function permits to remove the '_' at the end of the labels.
    :param char_list:
    :return: the character without the '_' at the end
    ex: 'London__________' -> 'London'
    """
    for index_char, char in enumerate(char_list):
        i = len(char)-1
        while char[i] == '_' and i > 0:
            char = char[:i]
            i -= 1
        char_list[index_char] = char
    return char_list


class Data:
    def __init__(self, images_dir_path, labels_txt_path, pad_input_char=True, normalize_pixels=True):

        self.lexic = []
        self.im_height = 28
        self.im_length = 384
        self.lb_length = 25  # normalized sequence length of the encoded elements of the target
        self.pad_input_char = pad_input_char
        self.image_dir_path = images_dir_path
        self.labels_txt_path = labels_txt_path

        self.labels_dict = self.get_labels_dict(labels_txt_path)
        self.encoding_dict = np.load(V.encoding_dict).item()
        self.decoding_dict = np.load(V.decoding_dict).item()

        self.images_path = np.array(os.listdir(images_dir_path))

        self.vocab_size = len(self.encoding_dict)  # total numbers of different characters within the vocabulary, (28)

        self.normalize_pixel = normalize_pixels
        self.pixel_mean = 49e-3         # cf utils/meanAndVar
        self.pixel_var = 20e-2

    def get_labels_dict(self, labels_txt_path):
        """
        :param labels_txt_path:
        :return: a dictionary of corresponce between image files and its associated labels

        ex :    { 'NEAT_0-19word12561420170630155102_005_w005.png': 'NICOSIA',
                  'NEAT_0-19word12583420170703115206_001_w007.png': 'IDEAS',
                  'NEAT_0-19word12534620170629153942_005_w008.png': 'WRITINGS', ... }
        """
        labels_dict = {}
        txt = open(labels_txt_path).read()  # text contenant les labels et les noms des fichiers images
        n_index = [i for i, x in enumerate(txt) if x == "\n"]  # indices des retours a la ligne das le text

        line = txt[0:n_index[0]]

        for i in range(len(n_index) - 1):
            s = line.index(' ')
            t = line.index('/')
            word = line[s + 1:]

            labels_dict[line[t + 1:s]] = word

            if word not in self.lexic:
                self.lexic.append(word)

            d = n_index[i] + 1
            f = n_index[i + 1]
            line = txt[d:f]

        s = line.index(' ')
        t = line.index('/')
        labels_dict[line[t + 1:s]] = line[s + 1:]

        return labels_dict

    def generator(self, batch_size):
        images_path = self.images_path

        if self.normalize_pixel:
            m = self.pixel_mean
            v = self.pixel_var
        else:
            m = 0
            v = 1
        while True:
            try:
                if batch_size <= len(images_path):
                    batch_ids = random.sample(range(len(images_path)), batch_size)  # list of random ids
                else:
                    batch_ids = random.sample(range(len(images_path)), len(images_path))
                    batch_size = len(images_path)

                images_path_batch = [images_path[id_] for id_ in batch_ids]  # list of random image paths
                images_path = [image_path for id_, image_path in enumerate(images_path) if id_ not in batch_ids]
                # we remove the images in the current list not to use the same image twice

                variable_size_images_batch_list = [(mpimg.imread(  # images to be padded
                    self.image_dir_path + image_path_batch) - m) / v
                                                   for image_path_batch in images_path_batch]

                images_batch_list = pad_images(variable_size_images_batch_list, self.im_height, self.im_length)

                images_batch = np.array(images_batch_list).reshape(batch_size, self.im_length, self.im_height, 1)

                words_batch_list = [self.labels_dict[image_path]
                                    # list of variable-size non-encoded words
                                    for image_path in images_path_batch]

                if self.pad_input_char:
                    # the sequence of character will be padded with '_' which correspond to the last one-hot encoding
                    words_batch_list = pad_character(words_batch_list, self.lb_length)

                labels_batch = self.encode_label(words_batch_list)  # encode and normalize size

                if batch_size > 0:
                    assert (images_batch.shape == (batch_size, self.im_length, self.im_height, 1)) & \
                           (labels_batch.shape == (batch_size, self.lb_length, self.vocab_size))

                yield (images_batch, labels_batch)
            except Exception as e:
                print("catch exception")
                print(e)
                self.generator(batch_size)


    def encode_label(self, labels):
        """
        :param labels: list of strings of variable length.
        :return: np.array representing a sequence of encoded elements.
        the sequences are completed with zero vectors.

        encoded_labels.shape = (batch_size, self.lb_length, self.vocab_size)
        """
        variable_size_encoded_labels_list = [[self.encoding_dict[char]
                                              for char in label] for label in labels]

        # the fallowing portion ensure that the encoded label has a fixed sized by padding it with zero vectors.
        # If pad_input_char == True the character sequence is already padded with '_' and the label will not contain any zero vector
        # as it will be padded with the vector corresponding to '_'.
        encoded_labels = np.array(
            [xi + [list(np.zeros(self.vocab_size))] * (self.lb_length - len(xi))
             for xi in variable_size_encoded_labels_list])

        return encoded_labels

    def decode_labels(self, labels, depad=False, deep_lib='keras'):

        """
        :param labels: np.array of list representing a sequence of encoded labels.
        :param depad: removes '_'
        :param onehot_input: True in the input is encoded as a onehot vector, false if it is a list of indexes
        :return: the corresponding list of strings padded with "_" for null elements
        """
        decoded_labels = []
        for label in labels:
            decoded_label = ''
            if deep_lib == 'keras':
                for e in label:
                    if np.sum(e) == 1:
                        decoded_label += self.decoding_dict[list(e).index(1.)]

            if deep_lib == 'pytorch':
                for e in label:
                    decoded_label += self.decoding_dict[e.item()]

            decoded_labels.append(decoded_label)

        if depad:
            decoded_labels = depad_character(decoded_labels)

        return decoded_labels


def main1():
    train = Data(V.images_train_dir, V.labels_train_txt)
    test = Data(V.images_test_dir, V.labels_test_txt)

    gen = train.generator(20)

    images, labels = gen.__next__()
    images, labels, labels_argmax = adapt_data_format(images, labels)

    print(images.shape)


if __name__ == "__main__":
    main1()

    print('fin')
