import torch
from torch_geometric.nn import MessagePassing

from math import sqrt


def init_mask(x, edge_index):
    (N, F), E = x.size(), edge_index.size(1)
    std = torch.nn.init.calculate_gain('relu') * sqrt(2.0 / (2 * N))
    return  torch.nn.Parameter(torch.randn(E, device=x.device) * std)

def set_mask(model, edge_mask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edge_mask

def clear_mask(model):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def meta_update_weights(model, params, verbose: bool=False):
    """Performs meta-update step of MATE meta-training, i.e. updates the models 
    parameters for the main calssification task w.r.t the adapted parameters 
    computed optimizing the explanation task.

    Args
    - `model`  : model for the main calssification task to be meta-updated;
    - `params` : params needed for the update;
    """
    # check that there are enough parameters for the update, i.e. 
    # that update parameters were computed correctly for the model 
    mod_params = len([child for child in model.children()])*2
    n_param    = len(params)
    if verbose:
        print("params  :", [par.size() for par in params])
        print("mod_params:", mod_params)
        print("n_params  :", n_param)
    assert mod_params == n_param, "Update parameters don't match model parameters!"

    idx = -n_param
    for name,mod in model.named_modules():
        if name == "": continue         # first elem of named_modules() is the model itslef
        if verbose: print(name,[t.size() for t in mod.parameters()])

        mod.weight.copy_(params[idx])
        mod.bias.copy_(params[idx+1])
        idx += 2

    return model

