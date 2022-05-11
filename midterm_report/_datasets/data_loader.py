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


def load_small(train_data_size=100):
    here = pathlib.Path(__file__).resolve().parent
    loc_train_data = here / 'train.pkl'
    loc_test_data = here / 'test.pkl'

    # load train/test data
    with open(loc_train_data, 'rb') as fin:
        train_data = np.array(pickle.load(fin), dtype=float)

    with open(loc_test_data, 'rb') as fin:
        test_data = np.array(pickle.load(fin), dtype=float)

    train_data_small = None
    for index in range(NUM_CLASSES):
        start_i = index * NUM_TRAIN_DATA_EACH_CLASS
        end_i = start_i + train_data_size
        train_sub = train_data[start_i:end_i, :]

        if train_data_small is None:
            train_data_small = train_sub
        else:
            train_data_small = np.concatenate((train_data_small, train_sub), axis=0)

    train_label = np.arange(1, NUM_CLASSES + 1)
    train_label = np.repeat(train_label, train_data_size, axis=0)

    test_label = np.arange(1, NUM_CLASSES + 1)
    test_label = np.repeat(test_label, NUM_TEST_DATA_EACH_CLASS, axis=0)

    return train_data_small, train_label, test_data, test_label

