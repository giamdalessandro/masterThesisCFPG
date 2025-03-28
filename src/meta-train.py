import os
import higher
import argparse
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.utils import k_hop_subgraph

from explainers.CFPGExplainer import CFPGExplainer
from explainers.PCFExplainer import PCFExplainer

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.evaluation import evaluate, store_checkpoint, load_best_model
from utils.graphs import normalize_adj
from utils.meta_learn import init_mask, clear_mask, set_mask, meta_update_weights


## TODO sistemare i parametri con argparse
#   e aggoirnare la versione di parse_config()
CUDA = True

parser = argparse.ArgumentParser()
parser.add_argument("--gnn", "-G", type=str, default='GNN')
parser.add_argument("--dataset", "-D", type=str, default='syn1')
parser.add_argument("--epochs", "-e", type=int, default=5, help='Number of explainer epochs.')
parser.add_argument("--seed", "-s", type=int, default=42, help='Random seed.')
parser.add_argument('--mode', type=str, default="meta")

parser.add_argument("--device", "-d", default="cpu", help="'cpu' or 'cuda'.")
parser.add_argument('--train', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--store', default=False, action=argparse.BooleanOptionalAction)

args = parser.parse_args()
#print(">>", args)
GNN_MODEL = args.gnn          # "GNN", "CF-GNN" or "PGE"
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
EPOCHS    = args.epochs       # explainer epochs
SEED  = args.seed
MODE  = args.mode              # "" for normal training, "adv" for adversarial
TRAIN = args.train
STORE = args.store
DEVICE = args.device


## STEP 1: load a BAshapes dataset
cfg = parse_config(dataset=DATASET, gnn=GNN_MODEL)
dataset, test_indices = load_dataset(dataset=DATASET)
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset.num_node_features})
idx_train = dataset.train_mask
idx_eval  = dataset.val_mask
idx_test  = dataset.test_mask


graph = dataset[0]
print(Fore.GREEN + f"[dataset]> {dataset} dataset graph...")
print("\t>>", graph)

labels = graph.y
labels = torch.argmax(labels, dim=1)
x = graph.x
edge_index = graph.edge_index #.indices()


#### STEP 2: instantiate GNN model
if GNN_MODEL == "CF-GNN":
    ## need dense adjacency matrix for GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    dense_edge_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    norm_edge_index = normalize_adj(dense_edge_index)

model, ckpt = model_selector(paper=cfg["paper"], dataset=DATASET, pretrained=not(TRAIN), config=cfg)


# extract explainer loss function for meta-train loop
print(Fore.MAGENTA + "\n[explain]> loading explainer...")
if GNN_MODEL == "GNN":
    explainer = CFPGExplainer(model, graph, coeffs=cfg["expl_params"])
elif GNN_MODEL == "CF-GNN":
    explainer = PCFExplainer(model, graph, norm_edge_index, coeffs=cfg["expl_params"])
expl_loss_fn = explainer.loss


"""
TODO Meta-training loop to be implemented
"""
#### STEP 3: Meta-training
train_params = cfg["train_params"]
expl_params = cfg["expl_params"]
if TRAIN:
    print(Fore.MAGENTA + "\n[meta-training]> starting train...")
    for p in model.parameters():
        p.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()
    meta_opt = torch.optim.Adam(model.parameters(), lr=train_params["lr"])      
    inner_opt = torch.optim.Adam(model.parameters(), lr=train_params["inner_lr"])

    best_val_acc = 0.0
    best_epoch = 0
    with tqdm(range(0, EPOCHS), desc="[meta-training]> Epoch") as epochs_bar:
        for epoch in epochs_bar:
            # extract a random node to train on
            idx = torch.randint(0, len(test_indices), (1,))
            node_idx = torch.tensor([test_indices[idx]]) 
            #print(Fore.BLUE + f"\n[testing]> Chosing node {node_idx.item()}...")

            # Extract computational subgraph
            if GNN_MODEL == "CF-GNN":
                sub_index = edge_idx = norm_edge_index
            else:
                sub_index = k_hop_subgraph(node_idx, 3, edge_index)[1]
                edge_idx = edge_index
            model.eval()

            if train_params["eval_enabled"]: model.eval()
            with torch.no_grad():       
                original_pred = model(x, sub_index)[node_idx]
                pred_label = original_pred.argmax(dim=1)
    
            # Compute explainer's parameters (K steps)
            edge_mask = init_mask(x, sub_index)
            opt_exp = torch.optim.Adam([edge_mask], lr=0.03)
            for _ in range(30):
                opt_exp.zero_grad()
                set_mask(model, edge_mask)
                masked_pred = model(x, sub_index)[node_idx]

                expl_loss = expl_loss_fn(masked_pred, pred_label, edge_mask)[0]                
                expl_loss.backward()
                opt_exp.step()       

            # Adapt model's parameters (T steps)
            model.train()
            meta_opt.zero_grad()
            edge_mask.requires_grad = False
            set_mask(model, edge_mask)
            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fnet, diffopt):
                for _ in range(train_params["T"]):
                    masked_pred = fnet(x, sub_index)[node_idx]
                    expl_loss = expl_loss_fn(masked_pred, pred_label, edge_mask)[0]
                    params = diffopt.step(expl_loss)


            # Meta-Update
            clear_mask(model)      
            with torch.no_grad():
                model = meta_update_weights(model, params, gnn=GNN_MODEL, verbose=False)

            out = model(x, edge_idx)

            loss = criterion(out[idx_train], labels[idx_train])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_params["clip_max"])
            meta_opt.step()
            
            if train_params["eval_enabled"]: model.eval()
            with torch.no_grad():
                out = model(x, edge_idx)


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
                        dataset=DATASET,
                        train_acc=train_acc, 
                        val_acc=val_acc, 
                        test_acc=test_acc, 
                        epoch=epoch,
                        mode=MODE) 

            if epoch - best_epoch > train_params["early_stop"] and best_val_acc > 0.99:
                break

    best_epoch = best_epoch if STORE else -1
    model = load_best_model(model=model, 
                best_epoch=best_epoch,
                gnn=GNN_MODEL, 
                dataset=DATASET, 
                mode=MODE,
                eval_enabled=train_params["eval_enabled"])
    out = model(x, edge_idx)

    # Train eval
    train_acc = evaluate(out[idx_train], labels[idx_train])
    test_acc  = evaluate(out[idx_test], labels[idx_test])
    val_acc   = evaluate(out[idx_eval], labels[idx_eval])
    print(Fore.MAGENTA + "\n[results]> training final results - Accuracy")
    if best_epoch == -1: print(Fore.RED+"[DEBUG]> training ckpts not stored, showing default results...")
    print(f"\t>> train: {train_acc:.4f}  val: {val_acc:.4f}  test: {test_acc:.4f}")

#### STEP 4: Store results
if STORE:
    store_checkpoint(
        model=model, 
        gnn=GNN_MODEL, 
        dataset=DATASET,
        train_acc=train_acc, 
        val_acc=val_acc, 
        test_acc=test_acc,
        mode=MODE)
    #store_train_results(_paper, _dataset, model, train_acc, val_acc, test_acc, desc=desc, meta=False)
