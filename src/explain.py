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


SEED   = 42
EPOCHS = 20   # explainer epochs
#TRAIN  = True
STORE_ADV = False
DATASET   = "BAcommunity"  # "BAshapes"(syn1), "BAcommunities"(syn2)
GNN_MODEL = "CF-GNN"    # "GNN" or "CF-GNN"


rel_path = f"/configs/{GNN_MODEL}/{DATASET}.json"
cfg_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
cfg = parse_config(config_path=cfg_path)


#### STEP 1: load a BAshapes dataset
DATASET = cfg["dataset"]
dataset, test_idxs = load_dataset(dataset=DATASET)
train_idxs = dataset.train_mask
# add dataset info to config 
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset.num_node_features})

graph = dataset[0]
print(Fore.GREEN + f"[dataset]> {dataset} dataset graph...")
print("\t>>", graph)
class_labels = graph.y
class_labels = np.argmax(class_labels, axis=1)

x = graph.x
edge_index = graph.edge_index


#### STEP 2: instantiate GNN model, one of GNN or CF-GNN
if GNN_MODEL == "CF-GNN":
    # need dense adjacency matrix for GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    norm_adj = normalize_adj(dense_index)

model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=True, config=cfg)


#### STEP 3: select explainer
print(Fore.RED + "\n[explain]> ...loading explainer")
#explainer = PGExplainer(model, edge_index, x, epochs=EPOCHS)
if GNN_MODEL == "GNN":
    explainer = CFPGExplainer(model, edge_index, x, epochs=EPOCHS)
elif GNN_MODEL == "CF-GNN":
    explainer = PCFExplainer(model, edge_index, norm_adj, x, epochs=EPOCHS) # needs 'CF-GNN' model


#### STEP 4: train and execute explainer
# Initialize evalution modules for AUC score and efficiency
gt = (graph.edge_index,graph.edge_label)
auc_eval = AUCEvaluation(ground_truth=gt, indices=test_idxs)
inference_eval = EfficiencyEvluation()

# ensure all modules have the same seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

inference_eval.reset()

# prepare the explainer (e.g. train the mlp-model if it's parametrized like PGEexpl)
indices = torch.tensor(test_idxs)
explainer.prepare(indices=indices)


# actually explain GNN predictions for all test indices
inference_eval.start_explaining()
explanations = []
with tqdm(test_idxs[:], desc=f"[{explainer.expl_name}]> ...testing", miniters=1, disable=False) as test_epoch:
    for idx in test_epoch:
        graph, expl = explainer.explain(idx)
        explanations.append((graph, expl))

inference_eval.done_explaining()

# compute AUC score for computed explanation
print(Fore.RED + "\n[explain]> ...computing metrics on eplanations")
auc_score = auc_eval.get_score(explanations)
time_score = inference_eval.get_score(explanations)

print(Fore.RED + "[explain]> AUC score   :",f"{auc_score:.4f}")
print(Fore.RED + "[explain]> time_elapsed:",f"{time_score:.4f}")

cf_examples = explainer.cf_examples
print(Fore.RED + "[explain]>",f"{len(cf_examples.keys())}","test nodes with at least one CF example.")
#print(Fore.RED + "[explain]> cf ex. for nodes :",f"{explainer.cf_examples.keys()}")



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
        "edge_labels" : dataset[0].edge_label,
    }
    print("adv_features :", adv_node_feats.size()) 

    # store in a .pkl file the adv examples
    rel_path = f"/../datasets/pkls/{DATASET}_adv_{GNN_MODEL}.pt"
    save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
    torch.save(adv_data, save_path)
