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

from evaluations.AUCEvaluation import AUCEvaluation
from evaluations.EfficiencyEvaluation import EfficiencyEvluation

from gnns.CFGNNpaper.gcn import GCNSynthetic


SEED   = 42
EPOCHS = 20   # explainer epochs
TRAIN  = True
STORE  = False
DATASET   = "bashapes"  # "bashapes" (syn1)
GNN_MODEL = "CF-GNN"    # "GNN" or "CF-GNN"


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
print(Fore.RED + "\n[explain]> ...loading explainer")
#explainer = CFPGExplainer(model, edge_index, x, epochs=EPOCHS)
#explainer = PGExplainer(model, edge_index, x, epochs=EPOCHS)
explainer = PCFExplainer(model, edge_index, norm_adj, x, epochs=EPOCHS) # needs 'CF-GNN' model


## STEP 4: train and execute explainer
# Initialize evalution modules for AUC score and efficiency
gt = (graph.edge_index,graph.edge_label)
auc_eval = AUCEvaluation(ground_truth=gt, indices=test_idxs)
inference_eval = EfficiencyEvluation()

# ensure all modules have the same seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

inference_eval.reset()

# prepare the explainer (e.g. train the mlp model if it's parametrized like PGEexpl)
indices = torch.tensor(test_idxs)
explainer.prepare(indices=test_idxs)

# actually explain GNN predictions for all test indices
inference_eval.start_explaining()
explanations = []
with tqdm(test_idxs[:], desc="[explain]> ...testing", miniters=1, disable=False) as test_epoch:
    for idx in test_epoch:
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))

inference_eval.done_explaining()

print(Fore.RED + "\n[explain]> ...explainer evaluation")
auc_score = auc_eval.get_score(explanations)
time_score = inference_eval.get_score(explanations)

print(Fore.RED + "[explain]> AUC score   :",f"{auc_score:.4f}")
print(Fore.RED + "[explain]> time_elapsed:",f"{time_score:.4f}")
