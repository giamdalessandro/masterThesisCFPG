import os
import torch
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

from gnns.paper_GNN.GNN import NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic

path_to_saves = "/../../checkpoints/"
SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_saves


def string_to_model(paper: str, dataset: str, device: str, config):
    """
    Given a paper and a dataset return the cooresponding neural model needed for training.

    Args
    - `paper`: the paper who's classification model we want to use.
    - `dataset`: the dataset on which we wish to train. This ensures that the model in- and output are correct.
    - `config`: the dict containing config file parameters.
    
    Returns 
        `torch.nn.module` models.
    """
    # get some model parameters from config file
    n_feat  = config["num_node_features"]
    n_hid   = config["num_hidden"]
    n_class = config["num_classes"]

    if paper == "GNN":  # GNNExplainer gnn model
        if dataset in ['syn1','syn2','syn3','syn4']:
            # node classification datasets
            return NodeGCN(n_feat, n_class, device)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented for GNN explainer.")

    elif paper == "CF-GNN":  # CF-GNNExplainer gnn model
        # get GCNSynth model parameter from config file
        drop = config["dropout"]

        if dataset in ['syn1','syn2','syn3','syn4']:
            # node classification datasets
            return GCNSynthetic(n_feat,n_hid,n_hid,n_class,drop)
        else:
            raise NotImplementedError(f"Dataset {dataset} not implemented for CF explainer.")
    
    else:
        raise NotImplementedError(f"Model {paper} not implemented.")

def get_pretrained_checkpoint(model, paper: str, dataset: str, explainer: str):
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
        rel_path = f"{paper}/{dataset}/{model_name}"
    else:
        rel_path = f"meta/{paper}/{explainer}/{dataset}/{model_name}"

    print(Fore.CYAN + "[models]> ...loading checkpoint from",f"'checkpoints/{rel_path}'")

    checkpoint = torch.load(SAVES_DIR + rel_path)
    if paper == "CF-GNN_old":
        model.load_state_dict(checkpoint)
        print(Fore.CYAN + f"[models]> Model checkpoint weights for: {[k for k,v in checkpoint.items()]}")
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(Fore.CYAN + "[models]> This model obtained:",
            f"train_acc: {checkpoint['train_acc']:.4f}",
            f"val_acc: {checkpoint['val_acc']:.4f}",
            f"test_acc: {checkpoint['test_acc']:.4f}.")

    return model, checkpoint

def model_selector(paper: str, dataset: str, explainer: str="",  pretrained: bool=True, 
                    return_checkpoint: bool=False, device: str="cpu", config=None): 
    r""" 
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
    print(Fore.CYAN + "\n[models]: loaded model...\n", model)
    if pretrained:
        model, checkpoint = get_pretrained_checkpoint(model, paper, dataset, explainer)
        return model, checkpoint
    
    return model, None