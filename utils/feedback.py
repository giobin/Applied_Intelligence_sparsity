import codecs

import torch
import wandb
from torch.nn.utils import prune

from utils.pruning import DERIVED_BATCHNORMALIZATION_TENSORS


class Logger:
    '''
    Simple logger to file and console.
    '''

    def __init__(self, log_path):
        self.path = log_path

    def logging(self, s, print_=True, log_=True):
        if print_:
            print(s)
        if log_:
            with codecs.open(self.path, 'a+', "utf-8") as f_log:
                f_log.write(s + '\n')


def get_stats(parameters_to_prune):
    '''
    Get for all parameters the percentage of remaining params.
    :param parameters_to_prune: list of model's parameters
    :return: a list of strings.
    '''
    overall_remaining, overall_nelement, stats = [], [], []
    params = [p[0] for p in parameters_to_prune]
    param_modules = list(sorted(set(params), key=params.index))
    i = 0

    for m in param_modules:
        pruned = prune.is_pruned(m)
        if pruned:
            t = m.named_buffers()
            for name, tnsr in t:
                if name not in DERIVED_BATCHNORMALIZATION_TENSORS:
                    assert name[-4:] == 'mask'
                    remaining = torch.sum(tnsr).item()
                    nelem = tnsr.nelement()
                    stat = f'{m}.{name[:-5]} {remaining} out of {nelem} -> {100. * remaining / nelem :.2f}%'
                    stats.append(stat)
                    overall_remaining.append(remaining)
                    overall_nelement.append(nelem)
                    wandb_name = f'rp.{i}.{m}.{name[:-5]}'
                    rp = 100. * remaining / nelem
                    wandb.log({wandb_name: rp})
        else:
            t = m.named_buffers()
            for name, tnsr in t:
                if name not in DERIVED_BATCHNORMALIZATION_TENSORS:
                    wandb_name = f'rp.{i}.{m}.{name}'
                    rp = 100
                    wandb.log({wandb_name: rp})
        i += 1

    if len(overall_remaining) != 0:
        starting_elem_num = sum(overall_nelement)
        remaining_par = sum(overall_remaining)
        masks_nelem = f'GLOBAL param num (counting also shared params), as is summing up masks nelem {starting_elem_num}'
        final_remaining_par = f'GLOBAL remaining pars {remaining_par} (counting also shared params) -> {100. * remaining_par / starting_elem_num :.2f}%'
        stats.append(final_remaining_par)
        stats.append(masks_nelem)
    return stats
