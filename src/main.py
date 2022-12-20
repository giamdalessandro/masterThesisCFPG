import numpy as np
from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.utils import k_hop_subgraph

from utils.datasets import load_dataset
from utils.evaluation import evaluate, store_checkpoint, load_best_model

from gnns.paper_GNN.GNN import GraphGCN, NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic
from utils.models import model_selector


dataset_name = "syn1"
LR             = 0.003
EPOCHS         = 1500
EARLY_STOP     = 100
## GNNExpl model
GNN_MODEL    = "GNN"
CLIP_MAX       = 2.0
EVAL_ENABLED   = True
## CF-GNNExpl model
#GNN_MODEL    = "CF-GNN"
N_HIDDEN       = 20

## load a BAshapes dataset
dataset, test_indices = load_dataset(dataset=dataset_name)
num_classes = dataset.num_classes
num_node_features = dataset.num_node_features
idx_train = dataset.train_mask
idx_eval  = dataset.val_mask
idx_test  = dataset.test_mask


graph = dataset[0]
print(Fore.GREEN + f"[dataset]> {dataset} dataset graph...")
print("\t>>", graph)
labels = graph.y
labels = np.argmax(labels, axis=1)
#print("\t#nodes:", graph.num_nodes)
#print("\t#edges:", graph.num_edges)args

## extract a random node to train on
#idx = torch.randint(0, len(test_indices), (1,))
#node_idx = torch.tensor([test_indices[idx]]) 
#print(Fore.BLUE + f"\n[testing]> Chosing node {node_idx.item()}...")

x = graph.x
edge_index = graph.edge_index #.indices()
#_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_index)
#print("\tedge_index       :", edge_index.size())
#print("\tnode neighborhood:", sub_index.size())
#print("\tnode features    :", x.size())


### instantiate GNN model
#model = NodeGCN(num_features=num_node_features, num_classes=num_classes, device="cpu")
model, _ = model_selector(paper=GNN_MODEL, dataset=dataset_name, pretrained=True)

### need dense adjacency matrix for GCNSynthetic model
#v = torch.ones(edge_index.size(1))
#s = (graph.num_nodes,graph.num_nodes)
#edge_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
#model = GCNSynthetic(nfeat=num_node_features,nhid=N_HIDDEN,nout=N_HIDDEN,nclass=num_classes,dropout=0.0)

# Define graph
print(Fore.BLUE + "[training]> loading model...\n", model)
print("-----------------------------\n")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.CrossEntropyLoss()


best_val_acc = 0.0
best_epoch = 0
with tqdm(range(0, EPOCHS), desc="[training]> Epoch") as epochs_bar:
    for epoch in epochs_bar:
        model.train()
        optimizer.zero_grad()
        #if paper[:3] == "GCN":
        #    out = model(x, norm_adj)
        #elif paper[:3] == "GNN":
        out = model(x, edge_index)

        loss = criterion(out[idx_train], labels[idx_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_MAX)
        optimizer.step()

        #if args.eval_enabled: model.eval()
        with torch.no_grad():
            #if args.paper[:3] == "GCN":
            #    out = model(x, norm_adj)
            #elif args.paper[:3] == "GNN":
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
            #store_checkpoint(
            #    model=model, 
            #    gnn=GNN_MODEL, 
            #    paper="", 
            #    dataset=dataset_name,
            #    train_acc=train_acc, 
            #    val_acc=val_acc, 
            #    test_acc=test_acc, 
            #    epoch=epoch)

        if epoch - best_epoch > EARLY_STOP and best_val_acc > 0.99:
            break

model = load_best_model(model, best_epoch, GNN_MODEL, "", dataset_name, EVAL_ENABLED)
#if args.paper[:3] == "GCN":
#    out = model(x, norm_adj)
#elif args.paper[:3] == "GNN":
out = model(x, edge_index)

# Train eval
train_acc = evaluate(out[idx_train], labels[idx_train])
test_acc  = evaluate(out[idx_test], labels[idx_test])
val_acc   = evaluate(out[idx_eval], labels[idx_eval])
print(Fore.MAGENTA + "[results]> training final results", 
        f"\n\ttrain_acc: {train_acc:.4f}",
        f"val_acc: {val_acc:.4f}",
        f"test_acc: {test_acc:.4f}")

#store_checkpoint(
#    model=model, 
#    gnn=GNN_MODEL, 
#    paper="", 
#    dataset=dataset_name,
#    train_acc=train_acc, 
#    val_acc=val_acc, 
#    test_acc=test_acc)
#store_train_results(_paper, _dataset, model, train_acc, val_acc, test_acc, desc=desc, meta=False)
