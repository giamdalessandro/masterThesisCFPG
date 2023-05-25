import os
import argparse
import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama
import torch

from utils.datasets import load_dataset, parse_config
from utils.models import model_selector
from utils.evaluation import evaluate, store_checkpoint, load_best_model 
from utils.graphs import normalize_adj


CUDA = True

parser = argparse.ArgumentParser()
parser.add_argument("--gnn", "-G", type=str, default='PGE',
                    choices=["PGE","CF-GNN"])
parser.add_argument("--dataset", "-D", type=str, default='syn1',
                    choices=["syn1","syn2","syn3","syn4"])
parser.add_argument("--epochs", "-e", type=int, default=0, help='Number of explainer epochs.')
parser.add_argument("--seed", "-s", type=int, default=42, help='Random seed.')
parser.add_argument('--adv', type=str, default="")

parser.add_argument('--train', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument('--store', default=False, action=argparse.BooleanOptionalAction)
parser.add_argument("--device", "-d", type=str, default="cpu", 
                    help="Running device.", choices=["cpu","cuda"])

args = parser.parse_args()
#print(">>", args)
DATASET   = args.dataset      # "BAshapes"(syn1), "BAcommunities"(syn2)
GNN_MODEL = args.gnn          # "GNN", "CF-GNN" or "PGE"
EPOCHS    = args.epochs       # explainer epochs
SEED  = args.seed
MODE  = args.adv              # "" for normal training, "adv" for adversarial
TRAIN = args.train
STORE = args.store

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

device = args.device
if torch.cuda.is_available() and device == "cuda" and CUDA:
    cuda_dev =  torch.cuda.device("cuda")
    print(">> cuda available", cuda_dev)
    print(">> device: ", torch.cuda.get_device_name(cuda_dev),"\n")


## load a BAshapes dataset
cfg = parse_config(dataset=DATASET, to_load=GNN_MODEL)
dataset, test_indices = load_dataset(dataset=DATASET)  #, load_adv=(MODE=="adv"))
graph = dataset.get(0) # get base BAgraph
print(Fore.GREEN + f"\n[dataset]> {dataset} dataset graph...")
print("\t>>", graph)

# add dataset info to config 
cfg.update({
    "num_classes": dataset.num_classes,
    "num_node_features": dataset[0].num_node_features})
idx_train = torch.BoolTensor(dataset.train_mask)
idx_eval  = torch.BoolTensor(dataset.val_mask)
idx_test  = torch.BoolTensor(dataset.test_mask)


### instantiate GNN modelgraph
model, _ = model_selector(paper=cfg["paper"], dataset=DATASET, pretrained=not(TRAIN), device=device, config=cfg)


#for g in range(dataset.len()):
#print(Fore.GREEN + f"\n[dataset]> {dataset} dataset graph...")
#print("\t>> current:", graph)

graph = dataset.get(0)    # get base BAgraph
labels = graph.y
labels = torch.argmax(labels, dim=1)
x = graph.x
edge_index = graph.edge_index #.indices()

if GNN_MODEL == "CF-GNN":
    ### need dense adjacency matrix forcuda available GCNSynthetic model
    v = torch.ones(edge_index.size(1))
    s = (graph.num_nodes,graph.num_nodes)
    edge_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
    edge_index = normalize_adj(edge_index)


if device == "cuda" and CUDA:
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


train_params = cfg["train_params"]
if TRAIN:
    print(Fore.MAGENTA + "\n[training]> starting train...")
    optimizer = torch.optim.Adam(model.parameters(), lr=train_params["lr"])#, weisght_decay=train_params["weight_decay"])
    #optimizer = torch.optim.SGD(model.parameters(), lr=train_params["lr"], nesterov=True, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    # training loop 
    best_val_acc = 0.0
    best_epoch = 0
    eps = train_params["epochs"] if EPOCHS == 0 else EPOCHS
    with tqdm(range(0, eps), desc=">> Epoch") as epochs_bar:
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

    out = model(x, edge_index)

    # Train eval
    train_acc = evaluate(out[idx_train], labels[idx_train])
    val_acc   = evaluate(out[idx_eval], labels[idx_eval])
    test_acc  = evaluate(out[idx_test], labels[idx_test])
    if STORE:
        # add metrics for rm-1hop and rm-expl test
        print(Fore.MAGENTA + "\n[results]> training results - Accuracy")
        print(f"\t>> model: {GNN_MODEL}\tdataset: {DATASET}")
    else:
        print(Fore.MAGENTA + "\n[results]> stored final results - Accuracy")
        if best_epoch == -1: print(Fore.RED+"[DEBUG]> training ckpts not stored, showing default results...")

    print(f"\t>> train: {train_acc:.4f}  val: {val_acc:.4f}  test: {test_acc:.4f}")


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
