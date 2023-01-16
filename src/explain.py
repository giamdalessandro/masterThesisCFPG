import os
import numpy as np
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from explainers.CFPGExplainer import CFPGExplainer
from explainers.PGExplainer import PGExplainer

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.evaluation import normalize_adj

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
#print(list(test_idxs))
#num_classes = dataset.num_classes
#num_node_features = dataset.num_node_features
#idx_train = dataset.train_mask
#idx_eval  = dataset.val_mask
#idx_test  = dataset.test_mask

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
    edge_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    edge_index = normalize_adj(edge_index)

model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=True, config=cfg)


## STEP 3: select explainer
#explainer = CFPGExplainer(model, edge_index, x, task="node", epochs=50)
explainer = PGExplainer(model, edge_index, x, task="node", epochs=50)
# prepare the explainer (e.g. train the mlp model if it's prametrized like PGEexpl)
explainer.prepare(indices=test_idxs)

## STEP 4: run experiment