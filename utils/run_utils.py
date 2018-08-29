import torch
import numpy as np


def adapt_data_format(x, y):
    x_adapt = torch.Tensor(x).permute(0, 3, 1, 2)
    y_adapt = np.argmax(y, axis=2)
    y_adapt = torch.LongTensor(y_adapt, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    return x_adapt, y_adapt


def save_loss(loss, epoch, advance, dataset):

    return None
