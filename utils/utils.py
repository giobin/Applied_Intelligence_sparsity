import os
import random
import shutil
from functools import partial
from glob import glob

import numpy
import torch
from tensor2tensor.utils.bleu_hook import uregex
from tokenizers import SentencePieceBPETokenizer, Tokenizer
from torch.utils.data import SequentialSampler, DataLoader
from torchnlp.samplers import BucketBatchSampler
from transformers import PreTrainedTokenizerFast

from utils.costants import *


def build_vocab(path, workdir, name, path_vocab=None):
    '''
    Build or load a vocab from a path.
    :param path: Path of files from where generate the vocabulary.
    :param workdir: Where to save the vocabulary.
    :param name: Name of the saved vocabulary.
    :param path_vocab: Path of already existent vocabulary. (Load vocabulary)
    :return:
    '''
    os.environ['TOKENIZERS_PARALLELISM'] = 'true'
    tokenizer = SentencePieceBPETokenizer()
    if path_vocab is None:
        tokenizer.train(files=[path], vocab_size=VOCAB_SIZE,
                        special_tokens=[PAD, BOS, EOS, USER_TAG_END, USER_TAG_START, AGENT_TAG_END, AGENT_TAG_START,
                                        NO_RESPONSE, UNK, CLS])

        tokenizer.save(workdir + "/" + name + ".vocab")
    else:
        path_vocab = os.path.abspath(os.getcwd()) + "/" + path_vocab
        tokenizer = Tokenizer.from_file(path_vocab)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    return PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token=PAD, eos_token=EOS, bos_token=BOS,
                                   unk_token=UNK)


def generate_batch(data_batch, l1_vocab, l2_vocab, args):
    '''
    From a batch of sentences, generate a truncated batch of tokens.
    :param data_batch: BAtch of sentences.
    :param l1_vocab: Vocabulary of the first language.
    :param l2_vocab: Vocabulary of the second language.
    :param args: args to decide to truncate from the left or from the right.
    :return: A Truncated batch of tokens.
    '''
    if args.type_dataset == "wmt":
        l1 = [truncate(elem[0], l1_vocab) for elem in data_batch]
        l2 = [truncate(elem[1], l2_vocab) for elem in data_batch]
    else:
        l1 = [truncate(elem[0], l1_vocab, left=True) for elem in data_batch]
        l2 = [truncate(elem[1], l2_vocab, left=True) for elem in data_batch]

    l1 = l1_vocab(l1, padding=True, return_tensors="pt")
    l2 = l2_vocab(l2, padding=True, return_tensors="pt")

    l1["labels"] = l2["input_ids"]
    l1['decoder_attention_mask'] = l2["attention_mask"]
    del l1['token_type_ids']
    return l1


def truncate(elem, tokenizer, left=False, add_special_tokens=True):
    '''
    Truncate a sentence.
    :param elem: Single sentence
    :param tokenizer: Vocabulary
    :param left: True if you want to truncate from the left.
    :param add_special_tokens:  True if you want to add special tokens (CLS, BOS, EOS)
    :return: a truncated sentence.
    '''
    truncated = False
    elem_coded = tokenizer(elem)['input_ids']
    if len(elem_coded) > MAX_LENGTH_TRUNCATION: truncated = True

    if left:
        elem_coded = elem_coded[-MAX_LENGTH_TRUNCATION:]
    else:
        elem_coded = elem_coded[:MAX_LENGTH_TRUNCATION]

    string = tokenizer.decode(elem_coded)

    if add_special_tokens:
        if not truncated:
            string = BOS + string
            string += EOS
        else:
            if left:
                string = CLS + string
                string += EOS
            else:
                string = BOS + string
                string += CLS
    return string


def get_configuration(args):
    '''
    :param args: Args
    :return: Model configuration.
    '''
    if args.type_dataset == "wmt":
        return MPE_wmt, EL_wmt, DL_wmt, FFN_DIM_wmt, AH_wmt, DM_wmt, DR_wmt
    return MPE_tsk1, EL_tsk1, DL_tsk1, FFN_DIM_tsk1, AH_tsk1, DM_tsk1, DR_tsk1


def get_param_optimizer(args):
    '''
    :param args: Args
    :return: Optimizer configuration.
    '''
    if args.type_dataset == "wmt":
        return BETA1_wmt, BETA2_wmt, EPS_wmt
    return BETA1_tsk1, BETA2_tsk1, EPS_tsk1


def get_hyper(args):
    '''
    :param args: Args
    :return: Beam search configuration.
    '''
    if args.type_dataset != "wmt":
        beam_size = 6
        length_penalty = 1.2
        length = 200
    else:
        beam_size = 4
        length_penalty = 0.6
        length = MAX_LENGTH_TRUNCATION + 50
    return beam_size, length_penalty, length


def get_spec_tokens(tokenizer):
    '''
    :param tokenizer: Vocabulary.
    :return: token of BOS, PAD, EOS.
    '''
    return tokenizer(PAD)['input_ids'][0], tokenizer(BOS)['input_ids'][0], tokenizer(EOS)['input_ids'][0]


def init_seeds():
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)


def count_parameters(model, trainable=True):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


### DATA LOADERS ####

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def generate_iterator_bucket(dataset, l1_vocab, l2_vocab, args):
    '''
    Generate a dataloader BucketBatchSampler.
    :param dataset: Parallel Dataset.
    :param l1_vocab: First Vocabulary.
    :param l2_vocab: Second vocabulary.
    :param args: Args
    :return: a dataloader.
    '''
    sampler = SequentialSampler(dataset)
    bucket_sampler = BucketBatchSampler(
        sampler, args.batch_size, False, sort_key=lambda r: len(dataset[r][0]))

    return DataLoader(dataset, batch_sampler=bucket_sampler,
                      collate_fn=partial(generate_batch, l1_vocab=l1_vocab, l2_vocab=l2_vocab, args=args),
                      num_workers=args.numworkers, worker_init_fn=seed_worker)


def pruned(path):
    checkpoint = torch.load(path)
    for k in checkpoint.keys():
        if '_orig' in k:
            return checkpoint
    return None


def bleu_tokenize(string):
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def check_max_n_ckpt(path, patience):
    ckpts = glob(os.path.join(path, '*/'))
    latest_files = []
    if len(ckpts) > patience + 1:
        latest_files = sorted(ckpts, key=os.path.getctime)
        for f in latest_files[:-patience - 1]:
            shutil.rmtree(f, ignore_errors=True)
    return latest_files[: -patience - 1]
