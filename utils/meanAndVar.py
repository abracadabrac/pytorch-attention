from data.reader import Data
from data.vars import Vars

import numpy as np

V = Vars()

train = Data(V.images_train_dir, V.labels_train_txt, normalize_pixels=False)
test = Data(V.images_test_dir, V.labels_test_txt, normalize_pixels=False)

gen_train = train.generator(4000)
gen_test = test.generator(4000)

images_train, _ = gen_train.__next__()
images_test, _ = gen_test.__next__()

print("mean train %f" % np.mean(images_train))
print("mean test %f" % np.mean(images_test))
print("var train %f" % np.std(images_train))
print("var test %f" % np.std(images_test))