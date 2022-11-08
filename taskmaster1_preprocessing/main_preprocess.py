import os

from taskmaster.file_manager import *

'''
Generate a dataset WMT14-like starting from Taskmaster-1.

'''
if __name__ == '__main__':
    DATASET_PATH = "/data/raw/corpus"

    # Load Dataset
    path_dataset = os.path.abspath(os.getcwd()) + DATASET_PATH
    dataset = get_dataset_taskmaster(path_dataset)

    # Get ids for splitting the Dataset in train, dev, test.
    ID_PATH = "/data/raw"
    path_ids = os.path.abspath(os.getcwd()) + ID_PATH
    train_ids, dev_ids, test_ids = get_ids(path_ids)

    train = []
    dev = []
    test = []

    for utterance in dataset:
        if utterance.id in train_ids:
            train.extend(utterance.getSentencesWithContext())

        if utterance.id in dev_ids:
            dev.extend(utterance.getSentencesWithContext())

        if utterance.id in test_ids:
            test.extend(utterance.getSentencesWithContext())

    # Write in new files the train, dev, test set.
    WRITE_PATH = "/data/processed"
    path_write = os.path.abspath(os.getcwd()) + WRITE_PATH
    if not os.path.exists(WRITE_PATH):
        os.makedirs(path_write, exist_ok=True)
    write_file(path_write, train, 'train.L1', 'train.L2')
    write_file(path_write, dev, 'dev.L1', 'dev.L2')
    write_file(path_write, test, 'test.L1', 'test.L2')
