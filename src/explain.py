import os
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from explainers.PGExplainer import PGExplainer
from explainers.CFPGExplainer import CFPGExplainer
from explainers.PCFExplainer import PCFExplainer

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.graphs import normalize_adj
from utils.plots import plot_graph

from evaluations.AUCEvaluation import AUCEvaluation
from evaluations.EfficiencyEvaluation import EfficiencyEvluation


CUDA = True
SEED   = 42
EPOCHS = 5           # explainer epochs
TRAIN_NODES = False
STORE_ADV = False
DATASET   = "syn1"    # "BAshapes"(syn1), "BAcommunities"(syn2)
GNN_MODEL = "GNN"     # "GNN", "CF-GNN" or "PGE"

# ensure all modules have the same seed
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = "cpu"
if torch.cuda.is_available() and CUDA:
    device = torch.cuda.device("cuda")
    print(">> cuda available", device)
    print(">> device: ", torch.cuda.get_device_name(device),"\n")
    device = "cuda"
    

if DATASET == "syn1": data_cfg = DATASET + "_BAshapes"
elif DATASET == "syn2": data_cfg = DATASET + "_BAcommunities"
elif DATASET == "syn3": data_cfg = DATASET + "_treeCycles"
elif DATASET == "syn4": data_cfg = DATASET + "_treeGrids"

rel_path = f"/configs/{GNN_MODEL}/{data_cfg}.json"
cfg_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
cfg = parse_config(config_path=cfg_path)


#### STEP 1: load a BAshapes dataset
DATASET = cfg["dataset"]
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
    # need dense adjacency matrix for GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    norm_adj = normalize_adj(dense_index)

model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=True, config=cfg, device=device)


# loading tensors for CUDA computation 
if torch.cuda.is_available() and CUDA:
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
    for idx in test_epoch:
        graph, expl = explainer.explain(idx)

        #print("subg :", graph.size())
        #print("expl :", expl.size())
        #print("mask :", mask.size())
        if idx == test_idxs[-1]: plot_graph(graph, expl_weights=expl, n_idx=idx, show=True)
        
        explanations.append((graph, expl))
        #exit(0)

inference_eval.done_explaining()

# compute AUC score for computed explanation
print(Fore.MAGENTA + "\n[explain]> explanation metrics")
auc_score = auc_eval.get_score(explanations)
time_score = inference_eval.get_score(explanations)
print("\t>> final score:",f"{auc_score:.4f}")
print("\t>> time elapsed:",f"{time_score:.4f}")

if GNN_MODEL != "PGE":      # PGE does not produce CF examples
    cf_examples = explainer.cf_examples
    max_cf_ex = len(train_idxs)
    print(Fore.MAGENTA + "[explain]>","test nodes with at least one CF example:",f"{len(cf_examples.keys())}/{max_cf_ex}")
    perc_cf = (len(cf_examples.keys())/max_cf_ex)
    print("\t>> with CF:",f"{perc_cf*100:.2f} %")
    #print("\t>> w/o CF :",f"{(1-perc_cf)*100:.2f} %")



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
