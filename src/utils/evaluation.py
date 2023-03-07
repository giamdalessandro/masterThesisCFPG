import os
import torch
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

path_to_saves = "/../../checkpoints/"
SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_saves


def evaluate(out, labels):
    """
    Calculates the accuracy between the prediction and the ground truth.
    
    Args:
    - `out`   : predicted outputs of the explainer
    - `labels`: ground truth of the data
    
    Returns 
        accuracy score
    """
    preds = out.argmax(dim=1)
    correct = preds == labels
    acc = int(correct.sum()) / int(correct.size(0))
    return acc

def store_checkpoint(model, gnn: str, dataset: str, train_acc, val_acc, test_acc, 
                    epoch=-1, mode: str=""):
    """
    Store the model weights at a predifined location.

    Args
    - `model`     : the model who's parameters we whish to save;
    - `gnn  `     : str, the gnn model;
    - `dataset`   : str, the dataset;
    - `train_acc` : training accuracy obtained by the model;
    - `val_acc`   : validation accuracy obtained by the model;
    - `test_acc`  : test accuracy obtained by the model;
    - `epoch`     : the current epoch of the training process;
    """
    if mode != "":
        save_path = SAVES_DIR + f"{gnn}/{mode}/{dataset}" 
    else:
        save_path = SAVES_DIR + f"{gnn}/{dataset}" 

    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    checkpoint = {'model_state_dict': model.state_dict(),
                  'train_acc': train_acc,
                  'val_acc': val_acc,
                  'test_acc': test_acc}
    if epoch == -1:
        torch.save(checkpoint, os.path.join(save_path, f"best_model"))
    else:
        torch.save(checkpoint, os.path.join(save_path, f"epoch_{epoch}"))

def load_best_model(model, best_epoch: int, gnn: str, dataset: str, explainer: str="",
                    eval_enabled: bool=True, mode: str=""):
    """
    Load the model parameters from a checkpoint into a model
    
    Args
    - `model`       : the model who's parameters overide
    - `best_epoch`  : the epoch which obtained the best result. 
            Use -1 to chose the "best model"
    - `gnn`         : str, the gnn 
    - `dataset`     : str, the dataset
    - `eval_enabled`: wheater to activate evaluation mode on the model or not
    - `mode`(str) : on of {"adv", "meta", ""}
    
    Returns 
        model with paramaters taken from the checkpoint
    """
    #print(Fore.RED + "\n[results]> best result at epoch", best_epoch)
    to_load = "best_model" if best_epoch == -1 else f"epoch_{best_epoch}"

    if mode != "":
        checkpoint = torch.load(SAVES_DIR + f"{gnn}/{mode}/{dataset}/{to_load}")
    else:
        checkpoint = torch.load(SAVES_DIR + f"{gnn}/{dataset}/{to_load}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    #print(Fore.MAGENTA + "[results]: best model", model)
    if eval_enabled: model.eval()

    return model