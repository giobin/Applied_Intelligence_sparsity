import torch
import torch.nn.utils.prune as prune

DERIVED_BATCHNORMALIZATION_TENSORS = ['running_mean', 'running_var', 'num_batches_tracked']


def select_pruning_params(model):
    '''
    :param model: Neural Network.
    :return: LIst of parameters of the model.
    '''
    parameters_to_prune = []
    model_base = model.model
    for name, module in model_base.named_modules():
        if hasattr(model_base, name):
            m = getattr(model_base, name)
            parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)

        if name == "encoder" or name == "decoder":
            for name_1, module_1 in module.named_modules():
                if hasattr(module, name_1):
                    if name_1 == "layers":
                        parameters_to_prune = add_layer_levels(module_1, parameters_to_prune)

                    m = getattr(module, name_1)
                    parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)

    m = getattr(model, "lm_head")
    parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)
    return parameters_to_prune


def add_layer_levels(layer, parameters_to_prune):
    for name, module in layer.named_modules():
        if name != "" and "." not in name:
            for name_1, module_1 in module.named_modules():
                if hasattr(module, name_1) and "self_attn." not in name_1:
                    m = getattr(module, name_1)
                    parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)

                if name_1 == "self_attn" or name_1 == "encoder_attn":
                    for name_2, module_2 in module_1.named_modules():
                        if hasattr(module_1, name_2):
                            m = getattr(module_1, name_2)
                            parameters_to_prune += add_param_to_pruning(m, weight=True, bias=True)
    return parameters_to_prune


def add_param_to_pruning(module, weight=True, bias=True):
    parameters_to_prune = []
    if weight and hasattr(module, 'weight'):
        parameters_to_prune.append((module, 'weight'))
    if bias and hasattr(module, 'bias') and module.bias is not None:
        parameters_to_prune.append((module, 'bias'))
    return parameters_to_prune


def count_params_under_threshold(parameters_to_prune, threshold):
    count = 0
    param_modules = list(set([p[0] for p in parameters_to_prune]))
    for m in param_modules:
        pruned = prune.is_pruned(m)
        t = m.named_parameters()
        for name, tnsr in t:
            if pruned:
                assert name[-4:] == 'orig'
                count += tnsr[torch.abs(tnsr) < threshold].nelement()
            else:
                count += tnsr[torch.abs(tnsr) < threshold].nelement()
    return count


def gamma_decay(initial_gamma, current_step, step_num_for_epoch):
    '''
    Decay the regularization term.
    '''
    gamma = initial_gamma - (initial_gamma / step_num_for_epoch) * current_step
    if gamma < 0.:
        gamma = 0.
    return gamma


def get_regularizer(model, gamma=0.5, alpha=0.1):
    '''
    Calculate the regulation factor described in the paper.
    '''
    regularizer = 0
    for param in model.parameters():
        if param.requires_grad:
            # get the gradient
            d_p = torch.abs(param.grad.data)
            # get the param square elem-wise
            p_square = param ** 2
            sparsity_importance = torch.exp(-d_p / alpha)
            regularizer += gamma * torch.sum(sparsity_importance * p_square)
    return regularizer

def get_l1_regularizer(model, gamma=0.5):
    '''
        Calculate the L1 regulation factor.
    '''
    regularizer = 0
    for param in model.parameters():
        if param.requires_grad:
            regularizer += gamma * torch.sum(torch.abs(param))
    return regularizer

def get_l2_regularizer(model, gamma=0.5):
    '''
        Calculate the L2 regulation factor.
    '''
    regularizer = 0
    for param in model.parameters():
        if param.requires_grad:
            regularizer += gamma * torch.sum(param**2)
    return regularizer


def crop(parameters_to_prune, amount):
    '''
    :param parameters_to_prune: List of prunable parameters
    :param amount: percentage of parameters to be removed.
    '''
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return


def fix_pruning(parameters_to_fix):
    for p in parameters_to_fix:
        if prune.is_pruned(p[0]):
            prune.remove(*p)
    return


def sum_masks(parameters_to_prune):
    '''
    :param parameters_to_prune: List of parameters.
    :return: Number of still prunable weights.
    '''
    non_zero_params = 0
    param_modules = set()

    i = 0
    for p in parameters_to_prune:
        if i < len(parameters_to_prune) - 1:
            # The last is the LM, with weights shared with the first
            # Embed level. Do not count his weights.
            param_modules.add(p[0])
        i = i + 1

    param_modules = list(param_modules)
    for m in param_modules:
        pruned = prune.is_pruned(m)
        if pruned:
            t = m.named_buffers()
            for name, tnsr in t:
                # Bypass derived tensors in BatchNormalization
                if name not in DERIVED_BATCHNORMALIZATION_TENSORS:
                    assert name[-4:] == 'mask'
                    non_zero_params += torch.sum(tnsr).item()
        else:
            t = m.named_parameters()
            for name, tnsr in t:
                non_zero_params += tnsr.nelement()
    return non_zero_params
