import torch
from torch_geometric.datasets import BAShapes
from torch_geometric.utils import k_hop_subgraph

from utils.datasets import load_dataset

from gnns.paper_GNN.GNN import GraphGCN, NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic


## load a BAshapes dataset
dataset, test_indices = load_dataset(dataset="syn1")
num_classes = dataset.num_classes

graph = dataset[0]
print(f"[dataset]> {dataset} dataset graph...")
print("\t->", graph)
#print("\t#nodes:", graph.num_nodes)
#print("\t#edges:", graph.num_edges)

## extract a random node to train on
idx = torch.randint(0, len(test_indices), (1,))
node_idx = torch.tensor([test_indices[idx]]) 
print(f"\n[testing]> Chosing node {node_idx.item()}...")

x = graph.x
edge_idx = graph.edge_index.indices()
_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_idx)
print("\tedge_index       :", edge_idx.size())
print("\tnode neighborhood:", sub_index.size())
print("\tnode features    :", x.size())


## instantiate GNNs model
print(f"[testing]> Testing on GNN models...")
model = NodeGCN(num_features=10, num_classes=num_classes, device="cpu")
output = model(x, sub_index)[node_idx]
print("\tGNN    ->", output)

## need dense adjacency matrix for GCNSynthetic model
v = torch.ones(sub_index.size(1))
s = (graph.num_nodes,graph.num_nodes)
sub_index = torch.sparse_coo_tensor(indices=sub_index, values=v, size=s).to_dense()

model = GCNSynthetic(nfeat=10,nhid=20,nout=20,nclass=num_classes,dropout=0.0)
output = model(x, sub_index)[node_idx]
print("\tCF-GNN ->", output)
