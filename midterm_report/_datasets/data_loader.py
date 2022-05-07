import pickle
import numpy as np
import pathlib

NUM_TRAIN_DATA_EACH_CLASS = 3000
NUM_TEST_DATA_EACH_CLASS = 387
NUM_CLASSES = 15


def load():
    here = pathlib.Path(__file__).resolve().parent
    loc_train_data = here / 'train.pkl'
    loc_test_data = here / 'test.pkl'

    # load train/test data
    with open(loc_train_data, 'rb') as fin:
        train_data = np.array(pickle.load(fin), dtype=float)

    with open(loc_test_data, 'rb') as fin:
        test_data = np.array(pickle.load(fin), dtype=float)

    train_label = np.arange(1, NUM_CLASSES + 1)
    train_label = np.repeat(train_label, NUM_TRAIN_DATA_EACH_CLASS, axis=0)

    test_label = np.arange(1, NUM_CLASSES + 1)
    test_label = np.repeat(test_label, NUM_TEST_DATA_EACH_CLASS, axis=0)

    return train_data, train_label, test_data, test_label

load()