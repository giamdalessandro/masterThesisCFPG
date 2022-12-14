import torch
import os

from src.gnns.paper_GNN.GNN import NodeGCN
from src.gnns.paper_CFGNN.gcn import GCNSynthetic



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

def get_pretrained_path(paper, dataset):
    """ TODO -> modify into get_pretrained_model
    Given a paper and dataset loads the pre-trained model.

    Args
    - `paper`: the paper who's classification model we want to use.
    - `dataset`: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    
    Returns 
        The path (`str`) to the pre-trined model parameters.
    """
    # maybe wirte get_pretrained_model function
    if paper == "GCN_old":
        # to load CFExpl paper pretrained models
        model_name = f"gcn_3layer_{dataset}.pt"
    else:
        model_name = "best_model"

    path = f"./checkpoints/{paper}/{dataset}/{model_name}"
    return path

def model_selector(paper: str, dataset: str, pretrained=True, return_checkpoint=False, device: str="cpu", config=None):
    """
    Given a paper and dataset loads accociated model.

    Args
    - `paper`: the paper who's classification model we want to use.
    - `dataset`: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    - `pretrained`: whether to return a pre-trained model or not.
    - `return_checkpoint`: whether to return the dict contining the models parameters or not.

    Returns 
        `torch.nn.module` models and optionallly a dict containing it's parameters.
    """
    model = string_to_model(paper, dataset, device, config)
    checkpoint = None
    if pretrained:
        path = get_pretrained_path(paper, dataset)
        print(f"[models]> ...loading checkpoint from '{path}'")
        checkpoint = torch.load(path)
        if paper == "GCN_old":
            model.load_state_dict(checkpoint)
            print(f"[models]> Model checkpoint weights for: {[k for k,v in checkpoint.items()]}")
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[models]> This model obtained: Train Acc: {checkpoint['train_acc']:.4f}, Val Acc: {checkpoint['val_acc']:.4f}, Test Acc: {checkpoint['test_acc']:.4f}.")
    
    if return_checkpoint:
        return model, checkpoint
    
    return model