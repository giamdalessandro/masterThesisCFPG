import os
import argparse
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
import numpy as np

from utils.general import parser_add_args, cuda_device_check
from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.explaining import explainer_selector
from utils.plots import plot_graph, plot_expl_loss, plot_mask_density, plot_scatter_node_mask
from utils.storelog import store_expl_log

from evaluations.AUCEvaluation import AUCEvaluation
from evaluations.EfficiencyEvaluation import EfficiencyEvaluation
from evaluations.CFEvaluation import get_cf_metrics

THRES = 0.1
#-E CFPGv2 -D syn1 -e 100 --roc --reg-size 0.001 --reg-cf 2.0 --reg-ent 1.0 --opt Adam --heads 3 --hid-gcn 20 --add-att 1.0
#-E CFPGv2 -D syn2 -e 100 --roc --reg-size 0.001 --reg-cf 2.0 --reg-ent 0.5 --opt Adam --heads 5 --hid-gcn 20 --add-att 5.0

CUDA = True
# explainer training
parser = argparse.ArgumentParser()
parser = parser_add_args(parser)

args = parser.parse_args()
print(">>", args, "\n")
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
EXPLAINER = args.explainer    # "GNN", "CF-GNN" or "PGE"
EPOCHS    = args.epochs       # explainer epochs
SEED      = args.seed
PLOT      = args.plot_expl
TRAIN_NODES = args.train_nodes
STORE_ADV   = args.store_adv
STORE_LOG   = args.log
VERBOSE = args.verbose

# ensure all modules have the same seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = args.device
cuda_device_check(device, CUDA, VERBOSE)


#### STEP 1: load a BAshapes dataset
cfg = parse_config(dataset=DATASET, to_load=EXPLAINER)
dataset, train_idxs, test_idxs = load_dataset(dataset=DATASET, verbose=VERBOSE)
#train_idxs = dataset.train_mask
# add some dataset info to config 
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset.num_node_features})

graph = dataset.get(0)
if VERBOSE: print(Fore.GREEN + "[dataset]> data graph from",f"{dataset}")
#if VERBOSE: print("\t>>", graph)
class_labels = graph.y
class_labels = torch.argmax(class_labels, dim=1)
x = graph.x
edge_index = graph.edge_index


#### STEP 2: instantiate pretrained GNN model to be explained
if DATASET == "cfg_syn4": DATASET = "syn4"
model, ckpt = model_selector(paper=cfg["paper"], dataset=DATASET, pretrained=True, 
                            config=cfg, device=device, verbose=VERBOSE)

# loading tensors for CUDA computation 
if device == "cuda" and CUDA:
    if VERBOSE: print("\n>> loading tensors to cuda...")
    model = model.to(device)
    for p in model.parameters():
        p.to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = class_labels.to(device)
    if VERBOSE: print(">> DONE")


#### STEP 3: select explainer
cfg["expl_params"]["thres"] = THRES
cfg["expl_params"]["early_stop"] = args.early_stop
explainer = explainer_selector(cfg, model, graph, args, VERBOSE)


#### STEP 4: train and execute explainer
# Initialize evalution modules for AUC score and efficiency
gt = (graph.edge_index,graph.edge_label,graph.pn_labels)
auc_eval = AUCEvaluation(ground_truth=gt, indices=test_idxs)
inference_eval = EfficiencyEvaluation()
inference_eval.reset()

# prepare the explainer (i.e. train the Explanation Module)
if TRAIN_NODES:
    train_idxs = torch.argwhere(torch.Tensor(train_idxs))

train_idxs = test_idxs   # use only nodes that have an explanation ground truth
explainer.prepare(indices=train_idxs)  # actually train the explainer model


# Actually explain GNN predictions for all test indices
inference_eval.start_explaining()
explanations = []
with tqdm(test_idxs[:], desc=f"[{explainer.expl_name}]> testing", miniters=1, disable=True) as test_epoch:
    top_k = 12 if DATASET != "syn4" else 24
    top_k = 0 if EXPLAINER in ["1hop","perfEx"] else top_k
    verbose = False
    curr_id = 0
    n_tests = len(test_epoch)
    for idx in test_epoch:
        subgraph, expl = explainer.explain(idx)

        if (curr_id%(n_tests//5)) == 0: 
            plot_graph(subgraph, expl_weights=expl, n_idx=idx, e_cap=top_k, show=PLOT, verbose=verbose)
        elif idx == test_idxs[-1]: 
            plot_graph(subgraph, expl_weights=expl, n_idx=idx, e_cap=top_k, show=PLOT, verbose=verbose)
        
        explanations.append((subgraph, expl, idx))
        curr_id += 1

inference_eval.done_explaining()

#if VERBOSE: print("\n\t>> expl labels matrix:", explainer.correct_labels.size())
#if VERBOSE: print("\t>> correct expl labels :", explainer.correct_labels.sum())
#if VERBOSE: print("\t>> original expl labels:", dataset.get(0).edge_label.values().sum())


# Metrics: compute AUC score for computed explanation
if VERBOSE: print(Fore.MAGENTA + "\n[explain]> Explanation metrics")
auc_score, roc_gts, roc_preds = auc_eval.get_score(explanations)
time_score = inference_eval.get_score(explanations)
if VERBOSE: print("\t>> final score:",f"{auc_score:.4f}")
if VERBOSE: print("\t>> time elapsed:",f"{time_score:.4f}")


#### STEP 5: Logs and plots
# CF explanations data to log
if EXPLAINER != "PGEex":      # PGE does not produce CF examples
    if EXPLAINER == "CFPG": explainer.coeffs["heads"] = "n/a"    

    cf_metrics = get_cf_metrics(
                    edge_labels=graph.pn_labels,
                    explanations=explanations,
                    counterfactuals=explainer.test_cf_examples,
                    n_nodes=x.size(0),
                    thres=THRES,
                    verbose=VERBOSE)
    
    test_cf = explainer.test_cf_examples 
    train_cf = explainer.cf_examples if EXPLAINER not in ["1hop","perfEx"] else test_cf
    #max_cf = len(train_idxs)
    max_train_cf = len(train_idxs)
    max_test_cf = len(test_idxs)

    test_fnd = len(test_cf.keys())
    train_fnd = len(train_cf.keys())
    test_cf_perc = (test_fnd/max_test_cf)
    train_cf_perc = (train_fnd/max_train_cf)
    
    if True: 
        print(Fore.MAGENTA + "[metrics]>","Average results on all explained predictions")
        print(f"\t>> Fidelity (avg): {cf_metrics[0]:.4f}", f"  [w/ CF: test: {test_fnd}/{max_test_cf} ({test_cf_perc*100:.2f}%), train: {train_fnd}/{max_train_cf} ({train_cf_perc*100:.2f}%)]")
        #print(f"\t>> Fidelity (avg): {cf_metrics[0]:.4f}")
        #print(f"\t\t-- w/ CF: test: {test_fnd}/{max_cf} ({test_cf_perc*100:.2f}%), train: {train_fnd}/{max_cf} ({train_cf_perc*100:.2f}%)")
        print(f"\t>> Sparsity (avg): {cf_metrics[1]:.4f}")
        print(f"\t>> Accuracy (avg): {cf_metrics[2]:.4f}")
        print(f"\t>> explSize (avg): {cf_metrics[3]:.2f}")
else:
    # add some log info for log function    
    explainer.coeffs["lr"] = explainer.lr 
    explainer.coeffs["opt"] = args.opt      
    explainer.coeffs["reg_cf"] = "n/a"    


# performances plots
if args.roc:
    e_name = explainer.expl_name
    e_h = explainer.history
    em_logs = {} if EXPLAINER != "CFPGv2" else explainer.explainer_module.logs_d
    plot_expl_loss(
        expl_name=e_name,
        dataset=DATASET,
        losses=e_h["train_loss"],
        cf_num=e_h["cf_fnd"] if EXPLAINER != "PGEex" else [-1],
        cf_test=test_fnd,
        cf_tot=e_h["cf_tot"] if EXPLAINER != "PGEex" else -1,
        roc_gt=roc_gts,
        roc_preds=roc_preds
    )
    plot_mask_density(explanations, em_logs, DATASET, EPOCHS, thres=THRES)
    #plot_scatter_node_mask(explanations)

#exit("\n[DEBUGGONE]> sto a fixà i plot")


# store explanation results into a log file
if STORE_LOG:
    logs_d = {
        "seed"      : SEED,
        "epochs"    : EPOCHS,
        "conv"      : args.conv,
        "nodes"     : "train" if TRAIN_NODES else "test",
        "e_cfg"     : explainer.coeffs,
        "nlays"     : explainer.n_layers if EXPLAINER == "CFPGv2" else 1,
        "time"      : time_score,
        "AUC"       : auc_score,
        "cf_test"   : test_cf_perc if EXPLAINER != "PGEex" else -1.0,
        "cf_train"  : train_cf_perc if EXPLAINER != "PGEex" else -1.0,
        "cf_tot"    : max_test_cf if EXPLAINER != "PGEex" else "a",
        "cf_train_tot" : max_train_cf if EXPLAINER != "PGEex" else "a",
        "fnd_test"  : test_fnd if EXPLAINER != "PGEex" else "n",
        "fnd_train" : train_fnd if EXPLAINER != "PGEex" else "n",
        "fidelity"  : cf_metrics[0] if EXPLAINER != "PGEex" else "n/a",
        "sparsity"  : cf_metrics[1] if EXPLAINER != "PGEex" else "n/a",
        "accuracy"  : cf_metrics[2] if EXPLAINER != "PGEex" else "n/a",
        "explSize"  : cf_metrics[3] if EXPLAINER != "PGEex" else "n/a",
    }
    store_expl_log(explainer=EXPLAINER, dataset=DATASET, logs=logs_d, prefix=args.prefix)


#### STEP 6: build the node_features for the adversarial graph based on the cf examples 
if STORE_ADV:
    adv_node_feats = []
    for n_idx in range(x.size(0)):
        try:
            adv_f = test_cf[str(n_idx)]["feat"]
            adv_f = torch.nn.functional.max_pool1d(adv_f.unsqueeze(dim=0), 2)#.squeeze()
            adv_node_feats.append(adv_f)
        except KeyError:
            adv_node_feats.append(x[n_idx].unsqueeze(dim=0))

    adv_node_feats = torch.cat(adv_node_feats, dim=0)
    adv_data = {
        "node_feats" : adv_node_feats,
        "edge_index" : edge_index,
        "labels" : class_labels,
        "train_idxs" : dataset.train_mask,
        "eval_idxs" : dataset.val_mask,
        "test_idxs" : dataset.test_mask,
        #"edge_labels" : dataset[0].edge_label,
    }
    if VERBOSE: print("adv_features :", adv_node_feats.size()) 

    # store in a .pkl file the adv examples
    rel_path = f"/../datasets/pkls/{DATASET}_adv_train_{EXPLAINER}.pt"
    save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
    torch.save(adv_data, save_path)
