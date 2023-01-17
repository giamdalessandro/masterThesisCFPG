import os
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from explainers.CFPGExplainer import CFPGExplainer
from explainers.PGExplainer import PGExplainer
from explainers.PCFExplainer import PCFExplainer

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.graphs import normalize_adj

from gnns.CFGNNpaper.gcn import GCNSynthetic

TRAIN = True
STORE = False
DATASET   = "bashapes"
GNN_MODEL = "GNN"

rel_path = f"/configs/{GNN_MODEL}/{DATASET}.json"
cfg_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
cfg = parse_config(config_path=cfg_path)


## STEP 1: load a BAshapes dataset
DATASET = cfg["dataset"]
dataset, test_idxs = load_dataset(dataset=DATASET)
num_classes = dataset.num_classes
#print(list(test_idxs))

graph = dataset[0]
print(Fore.GREEN + f"[dataset]> {dataset} dataset graph...")
print("\t>>", graph)
labels = graph.y
labels = np.argmax(labels, axis=1)

x = graph.x
edge_index = graph.edge_index


## STEP 2: instantiate GNN model, one of GNN or CF-GNN
if GNN_MODEL == "CF-GNN":
    # need dense adjacency matrix for GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    norm_adj = normalize_adj(dense_index)

model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=True, config=cfg)

## STEP 3: select explainer
explainer = CFPGExplainer(model, edge_index, x, task="node", epochs=30)
#explainer = PGExplainer(model, edge_index, x, task="node", epochs=50)
#explainer = PCFExplainer(model, edge_index, norm_adj, x, task="node", epochs=50)

# prepare the explainer (e.g. train the mlp model if it's parametrized like PGEexpl)
explainer.prepare(indices=test_idxs)

## STEP 4: run experiment
explanations = []
with tqdm(test_idxs[:], desc="[replication]> ...testing indexes", miniters=1, disable=False) as test_epoch:
    for idx in test_epoch:
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))