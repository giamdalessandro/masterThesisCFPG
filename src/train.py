import os
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.utils import k_hop_subgraph

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.evaluation import evaluate, store_checkpoint, load_best_model 
from utils.graphs import normalize_adj

TRAIN = True
STORE = False
DATASET   = "syn3_treeCycles" #"BAshapes", "BAcommunities", treeGrids
GNN_MODEL = "CF-GNN"        # "GNN", "CF-GNN"

CUDA = True
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = "cpu"
if torch.cuda.is_available() and CUDA:
    device =  torch.cuda.device("cuda")
    print(">> cuda available", device)
    print(">> device: ", torch.cuda.get_device_name(device),"\n")
    device = "cuda"

rel_path = f"/configs/{GNN_MODEL}/{DATASET}.json"
cfg_path = os.path.dirname(os.path.realpath(__file__)) + rel_path
cfg = parse_config(config_path=cfg_path)


## load a BAshapes dataset
DATASET = cfg["dataset"]
dataset, test_indices = load_dataset(dataset=DATASET, load_adv=False)
# add dataset info to config 
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset[0].num_node_features})
idx_train = torch.BoolTensor(dataset.train_mask)
idx_eval  = torch.BoolTensor(dataset.val_mask)
idx_test  = torch.BoolTensor(dataset.test_mask)


graph = dataset.get(0)    # get base BAgraph
print(Fore.GREEN + f"[dataset]> {dataset} dataset graph...")
print("\t>>", graph)

labels = graph.y
#labels = torch.argmax(labels, dim=1)
labels = np.argmax(labels, axis=1)
#print(">>>>", labels.size())
x = graph.x
edge_index = graph.edge_index #.indices()

## extract a random node to train on
#idx = torch.randint(0, len(test_indices), (1,))
#node_idx = torch.tensor([test_indices[idx]]) 
#print(Fore.BLUE + f"\n[testing]> Chosing node {node_idx.item()}...")
#_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_index)
#print("\tedge_index       :", edge_index.size())
#print("\tnode neighborhood:", sub_index.size())
#print("\tnode features    :", x.size())

if GNN_MODEL == "CF-GNN":
    ### need dense adjacency matrix forcuda available GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    edge_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    edge_index = normalize_adj(edge_index)


### instantiate GNN modelgraph
model, ckpt = model_selector(paper=GNN_MODEL, dataset=DATASET, pretrained=not(TRAIN), device=device, config=cfg)

if torch.cuda.is_available() and CUDA:
    print(">> loading tensors to cuda...")
    model = model.to(device)
    for p in model.parameters():
        p.to(device)

    x = x.to(device)
    edge_index = edge_index.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_eval = idx_eval.to(device)
    idx_test = idx_test.to(device)
    print(">> DONE")


# Define graph
if TRAIN:
    print(Fore.RED + "\n[training]> starting train...")
    train_params = cfg["train_params"]
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"])#, weisght_decay=train_params["weight_decay"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lr"], nesterov=True, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop 
    best_val_acc = 0.0
    best_epoch = 0
    with tqdm(range(0, train_params["epochs"]), desc=">> Epoch") as epochs_bar:
        for epoch in epochs_bar:
            model.train()
            optimizer.zero_grad()
            out = model(x, edge_index)
            
            loss = criterion(out[idx_train], labels[idx_train])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_params["clip_max"])
            optimizer.step()

            if train_params["eval_enabled"]: model.eval()
            with torch.no_grad():
                out = model(x, edge_index)
    
            # Evaluate train
            train_acc = evaluate(out[idx_train], labels[idx_train])
            test_acc  = evaluate(out[idx_test], labels[idx_test])
            val_acc   = evaluate(out[idx_eval], labels[idx_eval])

            epochs_bar.set_postfix(loss=f"{loss:.4f}", train_acc=f"{train_acc:.4f}", 
                                val_acc=f"{val_acc:.4f}", best_val_acc=f"{best_val_acc:.4f}")

            if val_acc > best_val_acc: # New best results
                best_val_acc = val_acc
                best_epoch = epoch
                if STORE:
                    store_checkpoint(
                        model=model, 
                        gnn=GNN_MODEL, 
                        paper="", 
                        dataset=DATASET,
                        train_acc=train_acc, 
                        val_acc=val_acc, 
                        test_acc=test_acc, 
                        epoch=epoch)

            if epoch - best_epoch > train_params["early_stop"] and best_val_acc > 0.99:
                break

    model = load_best_model(model=model, 
                best_epoch=-1, #best_epoch,#-1,
                paper=GNN_MODEL, 
                dataset=DATASET, 
                eval_enabled=train_params["eval_enabled"])

    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[idx_train], labels[idx_train])
    test_acc  = evaluate(out[idx_test], labels[idx_test])
    val_acc   = evaluate(out[idx_eval], labels[idx_eval])
    print(Fore.RED + "[results]> training final results", 
            f"\n\ttrain_acc: {train_acc:.4f}",
            f"val_acc: {val_acc:.4f}",
            f"test_acc: {test_acc:.4f}")


if STORE:
    store_checkpoint(
        model=model, 
        gnn=GNN_MODEL, 
        paper="", 
        dataset=DATASET,
        train_acc=train_acc, 
        val_acc=val_acc, 
        test_acc=test_acc)
    #store_train_results(_paper, _dataset, model, train_acc, val_acc, test_acc, desc=desc, meta=False)
