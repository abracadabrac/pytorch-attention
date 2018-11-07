import torch
from torch.autograd import Variable
import numpy as np


def adapt_data_format(x, y):
    x_adapt = Variable(torch.Tensor(x), requires_grad=True).permute(0, 3, 1, 2)

    y_adapt_argmax = np.argmax(y, axis=2)
    y_adapt_argmax = Variable(torch.LongTensor(y_adapt_argmax, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                       requires_grad=True)

    y_adapt = Variable(torch.Tensor(y, device=torch.device("cuda" if torch.cuda.is_available() else "cpu")),
                       requires_grad=True)

    return x_adapt, y_adapt, y_adapt_argmax
