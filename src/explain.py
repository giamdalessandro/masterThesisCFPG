import os
import argparse
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from explainers.PGExplainer import PGExplainer
from explainers.CFPGExplainer import CFPGExplainer
from explainers.CFPGv2 import CFPGv2
from explainers.OneHopExplainer import OneHopExplainer, PerfectExplainer
#from explainers.PCFExplainer import PCFExplainer
#from utils.graphs import normalize_adj

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.plots import plot_graph, plot_expl_loss
from utils.evaluation import store_expl_log, parser_add_args

from evaluations.AUCEvaluation import AUCEvaluation
from evaluations.EfficiencyEvaluation import EfficiencyEvaluation



CUDA = True
# explainer training
parser = argparse.ArgumentParser()
parser = parser_add_args(parser)

args = parser.parse_args()
print(">>", args)
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
EXPLAINER = args.explainer    # "GNN", "CF-GNN" or "PGE"
EPOCHS    = args.epochs       # explainer epochs
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
cfg = parse_config(dataset=DATASET, to_load=EXPLAINER)
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
#if EXPLAINER == "CF-GNN":
#    # need dense-normalized adjacency matrix for GCNSynthetic model
#    v = torch.ones(edge_index.size(1))
#    s = (graph.num_nodes,graph.num_nodes)
#    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
#    norm_adj = normalize_adj(dense_index)

if DATASET == "cfg_syn4": DATASET = "syn4"
model, ckpt = model_selector(paper=cfg["paper"], dataset=DATASET, pretrained=True, config=cfg, device=device)


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
cfg["expl_params"]["reg_ent"] = cfg["expl_params"]["reg_ent"] if args.reg_ent == 0.0 else args.reg_ent
cfg["expl_params"]["reg_size"] = cfg["expl_params"]["reg_size"] if args.reg_size == 0.0 else args.reg_size
cfg["expl_params"]["reg_cf"] = cfg["expl_params"]["reg_cf"] if args.reg_cf == 0.0 else args.reg_cf

if EXPLAINER == "PGEex":
    explainer = PGExplainer(model, graph, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"]) # needs 'GNN' model
elif EXPLAINER == "CFPG":
    explainer = CFPGExplainer(model, graph, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"])
elif EXPLAINER == "CFPGv2":
    conv = cfg["expl_params"]["conv"] if args.conv == "base" else args.conv
    cfg["expl_params"]["conv"]    = conv
    cfg["expl_params"]["heads"]   = args.heads
    cfg["expl_params"]["add_att"] = args.add_att
    explainer = CFPGv2(model, graph, conv=conv, epochs=EPOCHS, coeffs=cfg["expl_params"])
elif EXPLAINER == "1hop":
    explainer = OneHopExplainer(model, graph, device=device)
elif EXPLAINER == "perfEx":
    explainer = PerfectExplainer(model, graph, device=device)
#elif EXPLAINER == "CF-GNN":
#    explainer = PCFExplainer(model, graph, norm_adj, epochs=EPOCHS, device=device, coeffs=cfg["expl_params"]) # needs 'CF-GNN' model


#### STEP 4: train and execute explainer
# Initialize evalution modules for AUC score and efficiency
gt = (graph.edge_index,graph.edge_label)
auc_eval = AUCEvaluation(ground_truth=gt, indices=test_idxs)
inference_eval = EfficiencyEvaluation()
inference_eval.reset()

# prepare the explainer (e.g. train the mlp-model if it's parametrized like PGEexpl)
#print(">>>> test nodes:", indices.size())
if TRAIN_NODES:
    train_idxs = torch.argwhere(torch.Tensor(train_idxs))
else:                              
    train_idxs = test_idxs   # use only nodes that have an explanation ground truth
explainer.prepare(indices=train_idxs)  # actually train the explainer model


# Actually explain GNN predictions for all test indices
inference_eval.start_explaining()
explanations = []
with tqdm(test_idxs[:], desc=f"[{explainer.expl_name}]> testing", miniters=1, disable=False) as test_epoch:
    top_k = 12 if DATASET != "syn4" else 24
    top_k = 0 if EXPLAINER in ["1hop","perfEx"] else top_k
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


# Metrics: compute AUC score for computed explanation
print(Fore.MAGENTA + "\n[explain]> explanation metrics")
auc_score, roc_gts, roc_preds = auc_eval.get_score(explanations)
time_score = inference_eval.get_score(explanations)
print("\t>> final score:",f"{auc_score:.4f}")
print("\t>> time elapsed:",f"{time_score:.4f}")


#### STEP 5: Logs and plots
# CF explanations data to log
if EXPLAINER != "PGEex":      # PGE does not produce CF examples
    if EXPLAINER == "CFPG" :explainer.coeffs["heads"] = "n/a"    

    cf_examples = explainer.cf_examples
    found_cf_ex = len(cf_examples.keys())
    max_cf_ex = len(train_idxs)
    print(Fore.MAGENTA + "[explain]>","test nodes with at least one CF example")
    perc_cf = (found_cf_ex/max_cf_ex)
    print(f"\t>> with CF: {found_cf_ex}/{max_cf_ex}  ({perc_cf*100:.2f}%)")
else:
    # add some log info for log function    
    explainer.coeffs["lr"] = explainer.lr 
    explainer.coeffs["opt"] = "Adam"      
    explainer.coeffs["reg_cf"] = "n/a"    


# performances plots
if args.roc:
    e_name = explainer.expl_name
    e_h = explainer.history
    plot_expl_loss(
        expl_name=e_name,
        dataset=DATASET,
        losses=e_h["train_loss"],
        cf_num=e_h["cf_fnd"] if EXPLAINER != "PGEex" else [-1],
        cf_tot=e_h["cf_tot"] if EXPLAINER != "PGEex" else -1,
        roc_gt=roc_gts,
        roc_preds=roc_preds
    )
#exit("[DEBUGGONE]> sto a fixà i plot")


# store explanation results into a log file
if STORE_LOG:
    logs_d = {
        "seed"    : SEED,
        "epochs"  : EPOCHS,
        "conv"    : args.conv,
        "cfg"     : explainer.coeffs,
        "nodes"   : "train" if TRAIN_NODES else "test",
        "AUC"     : auc_score,
        "time"    : time_score,
        "cf_perc" : perc_cf if EXPLAINER != "PGEex" else -1.0,
        "cf_tot"  : max_cf_ex if EXPLAINER != "PGEex" else "a",
        "cf_fnd"  : found_cf_ex if EXPLAINER != "PGEex" else "n",
    }
    store_expl_log(explainer=EXPLAINER, dataset=DATASET, logs=logs_d, prefix=args.prefix)


#### STEP 6: build the node_features for the adversarial graph based on the cf examples 
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
    rel_path = f"/../datasets/pkls/{DATASET}_adv_train_{EXPLAINER}.pt"
    save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
    torch.save(adv_data, save_path)
