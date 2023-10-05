import os
import torch
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
    model : `torch.nn.Module`
        the model who's parameters we whish to save;
    gnn : `str` 
        the gnn model;
    dataset : `str`
        the dataset;
    train_acc : `float`
        training accuracy obtained by the model;
    val_acc : `float`
        validation accuracy obtained by the model;
    test_acc : `float`
        test accuracy obtained by the model;
    epoch : `int` 
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
    
    #### Args
    model : `torch.nn.Module` 
        the gnn model for which we wish to load the best result params
    best_epoch : `int` 
        the epoch which obtained the best result (Use -1 to chose the "best model")
    gnn : `str`
        the gnn name, for path
    dataset : `str`
        the dataset, for path
    eval_enabled : `bool`
        wheater to activate evaluation mode on the model or not
    mode : `str` 
        one of {"adv", "meta", ""}
    
    #### Returns 
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

def store_expl_checkpoint(model, dataset: str, epoch: int):
    """Store explainer model checkpoint into saves directory"""
    expl = model.expl_name
    save_path = SAVES_DIR + f"{expl}/{dataset}" 
    if not os.path.isdir(save_path):
        os.makedirs(save_path)

    if expl == "CFPGv2":
        expl_model = model.explainer_module
        checkpoint = {"model_state_dict": expl_model.state_dict(),
                    "dataset": dataset,
                    "epoch": epoch,
                    "conv": model.conv,
                    "thres": model.thres}
        
        save_name = f"{model.conv}{model.n_layers}_thres{model.thres}_e{epoch:03}"
    else:    
        expl_model = model.explainer_mlp
        checkpoint = {"model_state_dict": expl_model.state_dict(),
                    "dataset": dataset,
                    "epoch": epoch}
        
        save_name = f"{dataset}_e{epoch}"

    if epoch == -1: torch.save(checkpoint, os.path.join(save_path, save_name+"_best"))
    else: torch.save(checkpoint, os.path.join(save_path, save_name))
    print("\n[log]> explainer checkpoint stored at", f"checkpoints/{expl}/{dataset}/")

    return

def load_expl_checkpoint(model, dataset: str, best_epoch: int):
    expl = model.expl_name
    to_load = "_best" if best_epoch == -1 else f"e{best_epoch:03}"

    path = SAVES_DIR + f"{expl}/{dataset}/"
    for f in os.listdir(path):
        if f[-5:] == to_load:
            checkpoint = torch.load(path + f)
            break
        elif f[-4:] == to_load:
            checkpoint = torch.load(path + f)
            break
    
    if expl == "CFPGv2": expl_model = model.explainer_module
    else: expl_model = model.explainer_mlp
        
    expl_model.load_state_dict(checkpoint['model_state_dict'])
    print("\n[log]: explainer checkpoint loaded from", f"{f}")

    return model



path_to_logs = "/../../logs/"
LOG_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_logs

def store_expl_log(explainer: str, dataset: str, logs: dict, prefix: str="", save_dir: str=LOG_DIR):
    """Store explanation run logs, both in a log file and as a .csv"""
    save_dir = save_dir + f"{explainer}/{dataset}" 
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        header = True
    else:
        header = False

    eps = logs["epochs"]
    opt = logs["e_cfg"]["opt"]
    conv = logs["conv"]
    nlays = logs["nlays"]
    log_file = f"{prefix}{explainer}_{dataset}_e{eps}_{conv}_{opt}.log"
    
    e_c = logs["e_cfg"]
    heads = "n/a" if conv in ["GCN","base"] else e_c["heads"]     # no meaning if using GCNconv

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
        log_f.write(f">> cf explanation found: {logs['fnd_test']}/{logs['cf_tot']} ({logs['cf_test']:.4f})\n\n")
        log_f.close()

    # balanced metric for AUC and cf%
    metric = (((1-logs["fidelity"])*0.34) + (logs["sparsity"]*0.33) + (logs["accuracy"]*0.33))

    date_csv = datetime.now().strftime("%d-%m_%H:%M")
    to_csv = {
        "run_id"    : [date_csv],
        "seed"      : [logs["seed"]],
        "explainer" : [explainer],
        "epochs"    : [eps],
        "dataset"   : [dataset],
        "expl_arch" : [f"{nlays}{conv}{e_c['hid_gcn']}->FC64->relu->FC1"] if explainer == "CFPGv2" else [explainer],
        "conv"      : [conv],
        "optimizer" : [opt],
        "l_rate"    : [e_c['lr']],
        "heads"     : [heads],
        "note"      : [prefix],
        "reg_ent"   : [e_c['reg_ent']],
        "reg_cf"    : [e_c['reg_cf']],
        "reg_size"  : [e_c['reg_size']],
        "AUC"       : [round(logs['AUC'],4)],
        "cf_train"  : [round(logs['cf_train'],4)],
        "fnd_train" : [f"{logs['fnd_train']}/{logs['cf_train_tot']}"],
        "cf_test"   : [round(logs['cf_test'],4)],
        "fnd_test"  : [f"{logs['fnd_test']}/{logs['cf_tot']}"],
        "fidelity"  : [round(logs["fidelity"],4)],
        "sparsity"  : [round(logs["sparsity"],4)],
        "accuracy"  : [round(logs["accuracy"],4)],
        "explSize"  : [round(logs["explSize"],2)],
        "metric"    : [round(metric,4)] if explainer != "PGEex" else ["n/a"],
    }
    df = pd.DataFrame.from_dict(to_csv)
    save_dir = LOG_DIR + f"{explainer}"
    csv_path = save_dir + f"/{explainer}_{dataset}.csv"
    df.to_csv(csv_path, mode="a+", header=not os.path.exists(csv_path), index=False)

    return

