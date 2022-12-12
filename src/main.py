import torch
from torch_geometric.datasets import BAShapes
from torch_geometric.utils import k_hop_subgraph

from gnns.paper_GNN.GNN import GraphGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic


## load a BAshapes dataset
dataset = BAShapes(connection_distribution="random")
print(f"[dataset]> ...loading dataset '{dataset}' from PyG")

print("\t#entries:      ", len(dataset))
print("\t#classes:      ", dataset.num_classes)
print("\t#node_features:", dataset.num_node_features)
print("\t#edge_features:", dataset.num_edge_features)

graph = dataset[0]
print(f"\n[dataset]> {dataset} dataset graph...")
print("\t->", graph)
print("\t#nodes:", graph.num_nodes)
print("\t#edges:", graph.num_edges)

## extract a random node to train on
l = range(400, 700, 5)
idx = torch.randint(0, len(l), (1,))
node_idx = torch.tensor([l[idx]]) 
print(f"Chosing node {node_idx.item()}...")

edge_idx = graph.edge_index
_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_idx)
print("\tnode neighborhood:", sub_index.size())

## node features
x = graph.node_stores[0]["x"]
print("\tnode features    :", x.size())

## instantiate a GNN model
model = GraphGCN(num_features=10, num_classes=4, device="cpu")
#model = GCNSynthetic(nfeat=10,nhid=20,nout=20,nclass=4,dropout=0.0)
output = model(x, sub_index)