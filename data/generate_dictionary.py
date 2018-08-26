import numpy as np
from data.reader import get_labels_dict


# you must use the valid set, it is the convention
def get_encoding_dicts(labels_dict):
    """
    :param labels_dict:
    :return: dictionary of chars and their corresponding one hot encoding in alphabetical order

    ex for a vocab of 3 letters : { 'a': [1, 0, 0],
                                    'b': [0, 1, 0],
                                    'c': [0, 0, 1] }
    """
    chars_list = []
    for word in labels_dict.values():
        for char in word:
            if char not in chars_list:
                chars_list.append(char)
    chars_list.sort()
    chars_list.append('_')

    encoding_dict = {}
    for index, char in enumerate(chars_list):
        char_encoded = list(np.zeros(len(chars_list)))
        char_encoded[index] = 1.
        encoding_dict[char] = char_encoded

    decoding_dict = {list(char_encoded).index(1): char
                     for (char, char_encoded) in encoding_dict.items()}

    return encoding_dict, decoding_dict


def Main_generate_Hamelin_encoding_dicts():
    root = "/Users/charles/Data/Hamelin/"
    images_valid_dir = root + "VAL/valid/"
    labels_valid_txt = root + "valid.txt"

    labels_dict = get_labels_dict(labels_valid_txt)
    encoding_dict, decoding_dict = get_encoding_dicts(labels_dict)

    np.save('encoding_dict_Hamelin', encoding_dict)
    np.save('decoding_dict_Hamelin', decoding_dict)


if __name__ == "__main__":

    Main_generate_Hamelin_encoding_dicts()
