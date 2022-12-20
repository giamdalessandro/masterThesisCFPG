import os
import torch
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

from gnns.paper_GNN.GNN import NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic



def string_to_model(paper: str, dataset: str, device: str, config):
    """ TODO
    Given a paper and a dataset return the cooresponding neural model needed for training.

    Args
    - `paper`: the paper who's classification model we want to use.
    - `dataset`: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    - `config`: the dict containing config file parameters.
    
    Returns 
        `torch.nn.module` models.
    """
    if paper[:3] == "GNN":
        if dataset in ['syn1']:
            return NodeGCN(10, 4, device)
        elif dataset in ['syn2']:
            return NodeGCN(10, 8, device)
        elif dataset in ['syn3']:
            return NodeGCN(10, 2, device)
        elif dataset in ['syn4']:
            return NodeGCN(10, 2, device)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented for GNN explainer.")

    elif (paper[:3] == "GCN") or (paper in ["CF","GNN_cfpg"]):
        # get GCNSynth model parameter from config file
        n_feat  = config.gcn.n_feat
        n_hid   = config.gcn.hidden
        n_class = config.num_classes
        drop    = config.gcn.dropout

        if dataset in ['syn1','syn2','syn3','syn4']:
            return GCNSynthetic(n_feat,n_hid,n_hid,n_class,drop)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented for CF explainer.")
    
    else:
        raise NotImplementedError

def get_pretrained_checkpoint(model, paper: str, explainer: str, dataset: str):
    """
    Given a paper and dataset loads the pre-trained model.

    Args
    - `model`     : model instance on which the pretrained checkpoint will be loaded.
    - `paper`     : the paper who's classification model we want to use.
    - `explainer` : the explainer model on which the gnn-model has been meta-trained 
            (if you want to load the model weights after meta-training).
    - `dataset` : the dataset on which we wish to train. This ensures that the model 
            input and output are correct.
    
    Returns 
        The path (`str`) to the pre-trined model parameters.
    """
    # maybe wirte get_pretrained_model function
    if paper == "CF-GNN_old":
        # to load CFExpl paper pretrained models
        model_name = f"gcn_3layer_{dataset}.pt"
    else:
        model_name = "best_model"

    if explainer == "":
        path = f"./checkpoints/{paper}/{dataset}/{model_name}"
    else:
        path = f"./checkpoints/meta/{paper}/{explainer}/{dataset}/{model_name}"

    print(f"\n[models]> ...loading checkpoint from '{path}'")

    checkpoint = torch.load(path)
    if paper == "CF-GNN_old":
        model.load_state_dict(checkpoint)
        print(f"[models]> Model checkpoint weights for: {[k for k,v in checkpoint.items()]}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"[models]> This model obtained: train_acc: {checkpoint['train_acc']:.4f}",
                f"val_acc: {checkpoint['val_acc']:.4f}",
                f"test_acc: {checkpoint['test_acc']:.4f}.")

    return model, checkpoint

def model_selector(paper: str, dataset: str, explainer: str="",  pretrained: bool=True, 
                    return_checkpoint: bool=False, device: str="cpu", config=None): 
    """
    Given a paper and dataset loads accociated model.

    Args
    - `paper`   : the paper who's classification model we want to use.
    - `dataset` : the dataset on which we wish to train. This ensures that the model
            input and output are correct.
    - `explainer` : the explainer model on which the gnn-model has been meta-trained 
            (if you want to load the model weights after meta-training).
    - `pretrained` : whether to return a pre-trained model or not.
    - `return_checkpoint`: whether to return the dict contining the models parameters or not.

    Returns 
        `torch.nn.module` models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset, device, config)
    if pretrained:
        model, checkpoint = get_pretrained_checkpoint(model, paper, dataset, explainer)
        return model, checkpoint
    
    print(Fore.BLUE + "[training]> new gnn model loaded")
    return model, None