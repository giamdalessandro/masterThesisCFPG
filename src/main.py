from tqdm import tqdm
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama
import numpy as np

import torch
#from torch_geometric.datasets import BAShapes
from torch_geometric.utils import k_hop_subgraph

from utils.datasets import load_dataset
from utils.evaluation import evaluate, store_checkpoint, load_best_model

from gnns.paper_GNN.GNN import GraphGCN, NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic


dataset_name = "syn1"
gnn_model    = "GNN"
lr             = 0.003
epochs         = 100
early_stopping = 10
clip_max       = 2.0
eval_enabled   = True


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
idx = torch.randint(0, len(test_indices), (1,))
node_idx = torch.tensor([test_indices[idx]]) 
print(Fore.BLUE + f"\n[testing]> Chosing node {node_idx.item()}...")

x = graph.x
edge_index = graph.edge_index #.indices()
_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_index)
print("\tedge_index       :", edge_index.size())
print("\tnode neighborhood:", sub_index.size())
print("\tnode features    :", x.size())


### instantiate GNNs model
#print(Fore.BLUE + "\n[testing]> Testing on GNN models...")
model = NodeGCN(num_features=num_node_features, num_classes=num_classes, device="cpu")
#output = model(x, sub_index)[node_idx]
#print("\tGNN    ->", output)
#
### need dense adjacency matrix for GCNSynthetic model
#v = torch.ones(sub_index.size(1))
#s = (graph.num_nodes,graph.num_nodes)
#sub_index = torch.sparse_coo_tensor(indices=sub_index, values=v, size=s).to_dense()
#
#model = GCNSynthetic(nfeat=num_node_features,nhid=20,nout=20,nclass=num_classes,dropout=0.0)
#output = model(x, sub_index)[node_idx]
#print("\tCF-GNN ->", output)


## load gcn model for chosen task and dataset
#model = model_selector(
#    paper=_paper, 
#    dataset=_dataset, 
#    pretrained=False, 
#    return_checkpoint=False, 
#    device=device, 
#    config=args).to(device)

# Define graph
print(Fore.BLUE + "\n[training]> loading model...\n", model)
print("-----------------------------\n")
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
criterion = torch.nn.CrossEntropyLoss()


best_val_acc = 0.0
best_epoch = 0
with tqdm(range(0, epochs), desc="[train]> Epoch") as epochs_bar:
    for epoch in epochs_bar:
        model.train()
        optimizer.zero_grad()
        #if paper[:3] == "GCN":
        #    out = model(x, norm_adj)
        #elif paper[:3] == "GNN":
        out = model(x, edge_index)

        loss = criterion(out[idx_train], labels[idx_train])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max)
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

        #print(f"Epoch: {epoch}, train_acc: {train_acc:.4f}, val_acc: {val_acc:.4f}, train_loss: {loss:.4f}")
        epochs_bar.set_postfix(loss=f"{loss:.4f}", train_acc=f"{train_acc:.4f}", 
                            val_acc=f"{val_acc:.4f}", best_val_acc=f"{best_val_acc:.4f}")

        if val_acc > best_val_acc: # New best results
            #print("Val improved")
            best_val_acc = val_acc
            best_epoch = epoch
            store_checkpoint(
                model=model, 
                gnn=gnn_model, 
                paper="", 
                dataset=dataset_name,
                train_acc=train_acc, 
                val_acc=val_acc, 
                test_acc=test_acc, 
                epoch=epoch)

        if epoch - best_epoch > early_stopping and best_val_acc > 0.99:
            break

model = load_best_model(model, best_epoch, gnn_model, "", dataset_name, eval_enabled)
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

store_checkpoint(
    model=model, 
    gnn=gnn_model, 
    paper="", 
    dataset=dataset_name,
    train_acc=train_acc, 
    val_acc=val_acc, 
    test_acc=test_acc)
#store_train_results(_paper, _dataset, model, train_acc, val_acc, test_acc, desc=desc, meta=False)
