import os
import torch
import argparse
import pandas as pd

from datetime import datetime
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama


path_to_saves = "/../../checkpoints/"
SAVES_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_saves

def evaluate(out, labels):
    """Calculates the accuracy between the prediction and the ground truth.
    
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
    """Store the model weights at a predifined location.

    #### Args
    model : torch.nn.Module
        the model who's parameters we whish to save;
    gnn : str, 
        the gnn model;
    dataset : str, 
        the dataset;
    train_acc : float
        training accuracy obtained by the model;
    val_acc : float
        validation accuracy obtained by the model;
    test_acc : float
        test accuracy obtained by the model;
    epoch : int 
        the current epoch of the training process;
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
    """Load the model parameters from a checkpoint into a model
    
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



path_to_logs = "/../../logs/"
LOG_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_logs

def parser_add_args(parser: argparse.ArgumentParser):
    """Add arguments to argparser."""
    parser.add_argument("--explainer", "-E", type=str, default="CFPG",
                    choices=["PGEex","CFPG","CFPGv2","1hop"])
    parser.add_argument("--dataset", "-D", type=str, default="syn1", 
                        choices=['syn1','syn2','syn3','syn4'], help="Dataset used for training")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of explainer epochs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train-nodes", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to explain original train nodes")

    # to test gnn conv, may move it to cfg.json
    parser.add_argument("--conv", "-c", type=str, default="base", 
                        choices=["base","GCN","GAT","pGCN","VAE"], help="Explainer graph convolution")
    parser.add_argument("--heads", type=int, default=1, help="Attention heads (if conv is 'GAT')")
    parser.add_argument("--add-att", type=float, default=0.0, help="Attention coeff")
    parser.add_argument("--reg-ent", type=float, default=0.0, help="Entropy loss coeff")
    parser.add_argument("--reg-size", type=float, default=0.0, help="Size loss coeff")
    parser.add_argument("--reg-cf", type=float, default=0.0, help="Pred loss coeff")

    # other arguments
    parser.add_argument("--prefix", type=str, default="")
    parser.add_argument("--log", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to store run logs")
    parser.add_argument("--roc", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to plot ROC curve")
    parser.add_argument('--plot-expl', default=False, action=argparse.BooleanOptionalAction, 
                        help="Plot some of the computed explanation")
    parser.add_argument("--store-adv", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to store adv samples")
    parser.add_argument("--device", "-d", default="cpu", help="Running device, 'cpu' or 'cuda'")

    return parser

def store_expl_log(explainer: str, dataset: str, logs: dict, prefix: str="", save_dir: str=LOG_DIR):
    """Store explanation run logs."""
    save_dir = save_dir + f"{explainer}/{dataset}" 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        header = True
    else:
        header = False

    eps = logs["epochs"]
    opt = logs["cfg"]["opt"]
    conv = logs["conv"]
    log_file = f"{prefix}{explainer}_{dataset}_e{eps}_{conv}_{opt}.log"
    
    e_c = logs["cfg"]
    heads = "n/a" if conv == "GCN" else e_c["heads"]     # no meaning if using GCNconv

    date_str = datetime.now().strftime("%d-%B_%H:%M")
    log_file = save_dir + "/" + log_file
    with open(log_file, "a+") as log_f:
        log_f.write(f"\n############################################################\n")
        log_f.write(f"---------- {explainer} - {dataset} - {date_str} ---------------\n")
        log_f.write(f">> epochs:     {eps} \t\tnode explained: {logs['nodes']}\n")
        log_f.write(f">> graph conv: {conv}\t\theads (if GAT): {heads}\n")
        log_f.write(f">> explModule: {conv}1->FC64->relu->FC1\n")
        log_f.write(f"\n---------- params ---------------------------------------\n")
        log_f.write(f">> lr:           {e_c['lr']}\t reg_ent:  {e_c['reg_ent']}\n")
        log_f.write(f">> temps:   {e_c['temps']}\t reg_cf:   {e_c['reg_cf']}\n")
        log_f.write(f">> sample bias:    {e_c['sample_bias']}\t reg_size: {e_c['reg_size']}\n")
        log_f.write(f"\n---------- results --------------------------------------\n")
        log_f.write(f">> AUC: {logs['AUC']:.4f}\t\t\t time elapsed: {logs['time']:.2f} s\n")
        log_f.write(f">> cf explanation found: {logs['cf_fnd']}/{logs['cf_tot']} ({logs['cf_perc']:.4f})\n\n")
        log_f.close()

    # balanced metric for AUC and cf%
    metric = (logs['AUC']*0.5) + (logs['cf_perc']*0.5)

    date_csv = datetime.now().strftime("%d-%m_%H:%M")
    to_csv = {
        "run_id"    : [date_csv],
        "seed"      : [logs["seed"]],
        "explainer" : [explainer],
        "dataset"   : [dataset],
        "epochs"    : [eps],
        "expl_arch" : [f"2{conv}50->FC64->relu->FC1"] if explainer == "CFPGv2" else [explainer],
        "heads"     : [heads],
        "note"      : [prefix],
        "metric"    : [round(metric,4)] if explainer != "PGEex" else ["n/a"],
        "AUC"       : [round(logs['AUC'],4)],
        "cf (%)"    : [round(logs['cf_perc'],4)],
        "cf tot."   : [f"{logs['cf_fnd']}/{logs['cf_tot']}"],
        "optimizer" : [opt],
        "l_rate"    : [e_c['lr']],
        "reg_ent"   : [e_c['reg_ent']],
        "reg_cf"    : [e_c['reg_cf']],
        "reg_size"  : [e_c['reg_size']],
    }
    df = pd.DataFrame.from_dict(to_csv)
    csv_path = save_dir + f"/{explainer}_{dataset}.csv"
    df.to_csv(csv_path, mode="a+", header=not os.path.exists(csv_path))

    return