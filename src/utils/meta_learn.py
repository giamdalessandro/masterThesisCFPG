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


def meta_update_weights(model, params, gnn: str, verbose: bool=False):
    """Performs meta-update step of MATE meta-training, i.e. updates the models 
    parameters for the main calssification task w.r.t the adapted parameters 
    computed optimizing the explanation task.

    ### Args
    model : `torch.nn.Module` 
        model for the main calssification task to be meta-updated;

    params : `torch.Tensor`
        updated params for the model;
    """
    # check that there are enough parameters for the update, i.e. 
    # that update parameters were computed correctly for the model 
    mod_params = len([child for child in model.children()])*2
    n_param    = len(params)
    if verbose:
        print("\n[DEBUG](meta_update_weight) model params")
        for mod in model.children():
            print([k for k in mod.state_dict().keys()],"->",[p.size() for k,p in mod.state_dict().items()])
        print("params    :", len(params))
        for par in params: print("\t", par.size()) 
        print("mod_params:", mod_params)
        print("n_params  :", n_param)
    #assert mod_params == n_param, "Update parameters don't match model parameters!" # not always true

    if gnn == "CF-GNN":
        idx = -n_param
        for name,mod in model.named_modules():
            if name == "": continue         # first elem of named_modules() is the model itslef
            if verbose: print(name,[t.size() for t in mod.parameters()])

            mod.weight.copy_(params[idx])
            mod.bias.copy_(params[idx+1])
            idx += 2
    
    elif gnn == "GNN":
        model.conv1.state_dict()["bias"].copy_(params[-9])
        model.conv1.state_dict()["lin.weight"].copy_(params[-7])
        model.conv2.state_dict()["bias"].copy_(params[-6])
        model.conv2.state_dict()["lin.weight"].copy_(params[-5])
        model.conv3.state_dict()["bias"].copy_(params[-4])
        model.conv3.state_dict()["lin.weight"].copy_(params[-3])
        model.lin.weight.copy_(params[-2])
        model.lin.bias.copy_(params[-1])

    return model

