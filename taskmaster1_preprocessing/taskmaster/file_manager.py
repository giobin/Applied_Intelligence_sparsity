import io
import json
from os import listdir
from os.path import isfile, join

from taskmaster.utterance import *


def get_dataset_taskmaster(path):
    '''
    Load taskmaster-1 raw corpus.
    :param path: Path of the raw Taskmaster-1 corpus.
    :return: A list of utterances.
    '''
    list_files = []
    for f in listdir(path):
        if isfile(join(path, f)) and f not in list_files:
            list_files.append(f)

    utterances = []

    for file in list_files:
        json_file = io.open(path + '/' + file, mode='r')
        string = json_file.read()
        json_ = json.loads(string)
        for conversation in json_:
            utterance = Utterance(conversation['utterances'], conversation['conversation_id'])
            utterances.append(utterance)

    return utterances


def get_ids(path_ids):
    '''
    :param path_ids: Path to the csv containing the ids for the train, dev, test dataset.
    :return: train_ids, dev_ids, test_ids
    '''
    train_ids = []
    dev_ids = []
    test_ids = []

    file = io.open(path_ids + '/' + 'train.csv', mode='r')
    for line in file:
        line = line.replace('\n', '')
        line = line.replace(',', '')
        train_ids.append(line)

    file = io.open(path_ids + '/' + 'dev.csv', mode='r')
    for line in file:
        line = line.replace('\n', '')
        line = line.replace(',', '')
        dev_ids.append(line)

    file = io.open(path_ids + '/' + 'test.csv', mode='r')
    for line in file:
        line = line.replace('\n', '')
        line = line.replace(',', '')
        test_ids.append(line)

    return train_ids, dev_ids, test_ids


def write_file(path_write, sentences, file_name1, file_name2):
    f1 = open(path_write + "/" + file_name1, "w")
    f2 = open(path_write + "/" + file_name2, "w")

    i = 0
    for sentence in sentences:
        f1.write(sentence[0].replace('\n', ' '))
        f2.write(sentence[1].replace('\n', ' '))
        if i < len(sentences) - 1:
            f1.write("\n")
            f2.write("\n")

        i += 1
    pass
