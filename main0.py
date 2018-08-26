from data.reader import Data
from data.vars import Vars

V = Vars()

if __name__ == '__main__':

    data = Data(V.images_train_dir, V.labels_train_txt)
    gen = data.generator(1, index_label=True)
    x = gen.__next__()

    print(x)

