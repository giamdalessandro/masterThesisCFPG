import os
import argparse
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from explainers.PGExplainer import PGExplainer
from explainers.CFPGExplainer import CFPGExplainer
from explainers.PCFExplainer import PCFExplainer
from explainers.CFPGv2 import CFPGv2

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.graphs import normalize_adj
from utils.plots import plot_graph
from utils.evaluation import store_expl_log

from evaluations.AUCEvaluation import AUCEvaluation
from evaluations.EfficiencyEvaluation import EfficiencyEvluation

CUDA = True

# explainer training
parser = argparse.ArgumentParser()
parser.add_argument("--explainer", "-E", type=str, default="GNN")
parser.add_argument("--dataset", "-D", type=str, default="syn1", 
                    help="One of ['syn1','syn2','syn3','syn4']")
parser.add_argument("--epochs", "-e", type=int, default=5, 
                    help="Number of explainer epochs")
parser.add_argument("--seed", "-s", type=int, default=42, 
                    help="Random seed (default: 42)")
parser.add_argument("--conv", "-c", type=str, default="GCN", 
                    help="Explainer graph convolution ('GCN' or 'GAT')")
parser.add_argument('--plot-expl', default=False, action=argparse.BooleanOptionalAction, 
                    help="Plot some of the computed explanation")

# other arguments
parser.add_argument("--device", "-d", default="cpu", help="Running device, 'cpu' or 'cuda'")
parser.add_argument("--train-nodes", default=False, action=argparse.BooleanOptionalAction,
                    help="Whether to explain original train nodes")
parser.add_argument("--log", default=False, action=argparse.BooleanOptionalAction, 
                    help="Whether to store run logs")
parser.add_argument("--store-adv", default=False, action=argparse.BooleanOptionalAction, 
                    help="Whether to store adv samples")
parser.add_argument("--roc", default=False, action=argparse.BooleanOptionalAction, 
                    help="Whether to plot ROC curve")

args = parser.parse_args()
print(">>", args)
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
GNN_MODEL = args.explainer    # "GNN", "CF-GNN" or "PGE"
EPOCHS    = args.epochs       # explainer epochs
CONV      = args.conv
SEED      = args.seed
PLOT      = args.plot_expl
TRAIN_NODES = args.train_nodes
STORE_ADV   = args.store_adv
STORE_LOG   = args.log

# ensure all modules have the same seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = args.device
if torch.cuda.is_available() and device == "cuda" and CUDA:
    cuda_dev = torch.cuda.device("cuda")
    print(">> cuda available", cuda_dev)
    print(">> device: ", torch.cuda.get_device_name(cuda_dev),"\n")
    

#### STEP 1: load a BAshapes dataset
cfg = parse_config(dataset=DATASET, gnn=GNN_MODEL)
dataset, test_idxs = load_dataset(dataset=DATASET)
train_idxs = dataset.train_mask
# add some dataset info to config 
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset.num_node_features})

graph = dataset.get(0)
print(Fore.GREEN + "[dataset]> data graph from",f"{dataset}")
print("\t>>", graph)
class_labels = graph.y
class_labels = torch.argmax(class_labels, dim=1)
x = graph.x
edge_index = graph.edge_index


#### STEP 2: instantiate GNN model, one of GNN or CF-GNN
if GNN_MODEL == "CF-GNN":
    # need dense-normalized adjacency matrix for GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    norm_adj = normalize_adj(dense_index)

model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=True, config=cfg, device=device)


# loading tensors for CUDA computation 
if device == "cuda" and CUDA:
    print("\n>> loading tensors to cuda...")
    model = model.to(device)
    for p in model.parameters():
        p.to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = class_labels.to(device)
    print(">> DONE")


#### STEP 3: select explainer
print(Fore.MAGENTA + "\n[explain]> loading explainer...")
#explainer = PGExplainer(model, edge_index, x, epochs=EPOCHS)
if GNN_MODEL == "GNN":
    explainer = CFPGExplainer(model, graph, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"])
elif GNN_MODEL == "CF-GNN":
    explainer = PCFExplainer(model, graph, norm_adj, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"]) # needs 'CF-GNN' model
elif GNN_MODEL == "PGE":
    explainer = PGExplainer(model, graph, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"]) # needs 'GNN' model
elif GNN_MODEL == "CFPGv2":
    explainer = CFPGv2(model, graph, conv=CONV, epochs=EPOCHS, coeffs=cfg["expl_params"])

#### STEP 4: train and execute explainer
# Initialize evalution modules for AUC score and efficiency
gt = (graph.edge_index,graph.edge_label)
auc_eval = AUCEvaluation(ground_truth=gt, indices=test_idxs)
inference_eval = EfficiencyEvluation()
inference_eval.reset()

# prepare the explainer (e.g. train the mlp-model if it's parametrized like PGEexpl)
#print(">>>> test nodes:", indices.size())
if TRAIN_NODES:
    train_idxs = torch.argwhere(torch.Tensor(train_idxs))
else:                              
    train_idxs = test_idxs   # use only nodes that have an explanation ground truth
explainer.prepare(indices=train_idxs)


# actually explain GNN predictions for all test indices
inference_eval.start_explaining()
explanations = []
with tqdm(test_idxs[:], desc=f"[{explainer.expl_name}]> testing", miniters=1, disable=False) as test_epoch:
    top_k = 12 if DATASET != "syn4" else 24
    verbose = False
    curr_id = 0
    n_tests = len(test_epoch)
    for idx in test_epoch:
        graph, expl = explainer.explain(idx)

        if (curr_id%(n_tests//5)) == 0: 
            plot_graph(graph, expl_weights=expl, n_idx=idx, e_cap=top_k, show=PLOT, verbose=verbose)
        elif idx == test_idxs[-1]: 
            plot_graph(graph, expl_weights=expl, n_idx=idx, e_cap=top_k, show=PLOT, verbose=verbose)
        
        explanations.append((graph, expl))
        curr_id += 1

inference_eval.done_explaining()

# compute AUC score for computed explanation
print(Fore.MAGENTA + "\n[explain]> explanation metrics")
auc_score = auc_eval.get_score(explanations, args.roc)
time_score = inference_eval.get_score(explanations)
print("\t>> final score:",f"{auc_score:.4f}")
print("\t>> time elapsed:",f"{time_score:.4f}")

if GNN_MODEL != "PGE":      # PGE does not produce CF examples
    cf_examples = explainer.cf_examples
    found_cf_ex = len(cf_examples.keys())
    max_cf_ex = len(train_idxs)
    print(Fore.MAGENTA + "[explain]>","test nodes with at least one CF example:")
    perc_cf = (found_cf_ex/max_cf_ex)
    print(f"\t>> with CF: {found_cf_ex}/{max_cf_ex}  ({perc_cf*100:.2f}%)")
    #print("\t>> w/o CF :",f"{(1-perc_cf)*100:.2f} %")
else:
    # add some log info for log function    
    explainer.coeffs["lr"] = explainer.lr 
    explainer.coeffs["opt"] = "Adam"      
    explainer.coeffs["reg_cf"] = "n/a"    

# store explanation results into a log file
if STORE_LOG:
    logs_d = {
        "epochs"  : EPOCHS,
        "conv"    : CONV,
        "cfg"     : explainer.coeffs,
        "nodes"   : "train" if TRAIN_NODES else "test",
        "AUC"     : auc_score,
        "time"    : time_score,
        "cf_perc" : perc_cf if GNN_MODEL != "PGE" else -1.0,
        "cf_tot"  : max_cf_ex if GNN_MODEL != "PGE" else "a",
        "cf_fnd"  : found_cf_ex if GNN_MODEL != "PGE" else "n",
    }
    store_expl_log(explainer=GNN_MODEL, dataset=DATASET, logs=logs_d)


#### STEP 5: build the node_features for the adversarial graph based on the cf examples 
if STORE_ADV:
    adv_node_feats = []
    for n_idx in range(x.size(0)):
        try:
            adv_f = cf_examples[str(n_idx)]["feat"]
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
    print("adv_features :", adv_node_feats.size()) 

    # store in a .pkl file the adv examples
    rel_path = f"/../datasets/pkls/{DATASET}_adv_train_{GNN_MODEL}.pt"
    save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
    torch.save(adv_data, save_path)
