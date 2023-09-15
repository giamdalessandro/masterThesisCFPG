import argparse
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch

from utils.models import model_selector
from utils.datasets import load_dataset, parse_config
from utils.storelog import load_expl_checkpoint, store_expl_checkpoint
from utils.explaining import explainer_selector
from utils.general import parser_add_args, cuda_device_check

from evaluations.CFEvaluation import get_cf_metrics


THRES = 0.1
parser = argparse.ArgumentParser()
parser = parser_add_args(parser)

args = parser.parse_args()
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
EXPLAINER = args.explainer    # "GNN", "CF-GNN" or "PGE"
EPOCHS    = args.epochs       # explainer epochs
SEED      = args.seed
TRAIN_NODES = args.train_nodes
VERBOSE = args.verbose

cfg = parse_config(dataset=DATASET, to_load=EXPLAINER)

## load dataset
dataset, test_idxs = load_dataset(dataset=DATASET, verbose=VERBOSE)
train_idxs = dataset.train_mask
cfg.update({"num_classes": dataset.num_classes, "num_node_features": dataset.num_node_features})

print(Fore.GREEN+"[dataset]>",f"data graph from '{DATASET}' as {dataset}")
graph = dataset.get(0)
class_labels = torch.argmax(graph.y, dim=1)


## load gnn model
model, ckpt = model_selector(paper=cfg["paper"], dataset=DATASET, pretrained=True, 
                            config=cfg, verbose=VERBOSE)

## load explainer
cfg["expl_params"]["thres"] = THRES
cfg["expl_params"]["early_stop"] = args.early_stop
explainer = explainer_selector(cfg, model, graph, args, VERBOSE)

explainer = load_expl_checkpoint(explainer, DATASET, -1)

## multi-seed testing
seeds = [42,64,112,156]
for s in seeds:
    torch.manual_seed(s)       # ensure all modules have the same seed
    torch.cuda.manual_seed(s)

    explanations = []
    for idx in (te := tqdm(test_idxs[:], desc=f"[{explainer.expl_name}]> testing", miniters=1)):
        subgraph, expl = explainer.explain(idx)   
        explanations.append((subgraph, expl, idx))

    #if EXPLAINER == "CFPG": explainer.coeffs["heads"] = "n/a"    
    cf_metrics = get_cf_metrics(
                    edge_labels=graph.pn_labels,
                    explanations=explanations,
                    counterfactuals=explainer.test_cf_examples,
                    n_nodes=graph.x.size(0),
                    thres=THRES,
                    verbose=VERBOSE)

    test_cf = explainer.test_cf_examples
    max_cf = len(test_idxs)

    test_fnd = len(test_cf.keys())
    test_cf_perc = (test_fnd/max_cf)
    
    print(f"\n[log]> using seed {s}...")
    print(Fore.MAGENTA+"[metrics]>","Average results on all explained predictions")
    print(f"\t>> Fidelity (avg): {cf_metrics[0]:.4f}",f"  [w/ CF: {test_fnd}/{max_cf} ({test_cf_perc*100:.2f}%)]")
    print(f"\t>> Sparsity (avg): {cf_metrics[1]:.4f}")
    print(f"\t>> Accuracy (avg): {cf_metrics[2]:.4f}")
    print(f"\t>> explSize (avg): {cf_metrics[3]:.2f}")
    print()