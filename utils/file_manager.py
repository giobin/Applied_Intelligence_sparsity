import io
from os import listdir
from os.path import isfile, join

import numpy as np


def get_dataset(path):
    '''
    :param path: Path to the directory
    :return: Parallel dataset of sentences.
    '''
    onlyfiles = list_of_files(path)
    return load_from_files(onlyfiles, path)


def list_of_files(path):
    '''
    :param path: Path of the directory
    :return: A list containing all the names of the files inside Path
    '''
    list_files = []
    for f in listdir(path):
        if isfile(join(path, f)) and f[:-3] not in list_files:
            list_files.append(f[:-3])

    return list_files


def file_len(fname):
    '''
    :param fname: Path to the file
    :return: Number of lines in the file
    '''
    with io.open(fname, encoding="utf-8") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_size(onlyfiles, path):
    '''
    :param onlyfiles:  List of files inside the directory.
    :param path: Path to the directory.
    :return: Total of sentences present in the dataset
    '''
    i = 0
    for file in onlyfiles:
        i += file_len(path + "/" + file + ".L1")

    return i


def load_from_files(onlyfiles, path):
    '''
    :param onlyfiles:  List of files inside the directory.
    :param path: Path to the directory.
    :return: Parallel dataset of sentences.
    '''
    dataset_size = get_size(onlyfiles, path)
    dataset = np.empty((dataset_size, 2), dtype=np.object)

    base = 0
    for file in onlyfiles:
        print("Loading " + file)
        f_en = io.open(path + "/" + file + ".L1", encoding="utf-8")
        f_de = io.open(path + "/" + file + ".L2", encoding="utf-8")

        offset = 0
        for line in f_en:
            line = line.replace('\n', '')
            dataset[base + offset][0] = line
            offset += 1

        offset = 0
        for line in f_de:
            line = line.replace('\n', '')
            dataset[base + offset][1] = line
            offset += 1

        base += offset

    return dataset
