import torch
from torch_geometric.datasets import BAShapes
from torch_geometric.utils import k_hop_subgraph

from utils.datasets import load_node_dataset

from gnns.paper_GNN.GNN import GraphGCN, NodeGCN
from gnns.paper_CFGNN.gcn import GCNSynthetic


## load a BAshapes dataset
#dataset = BAShapes(connection_distribution="random")

dataset = load_node_dataset(dataset="syn1")

graph = dataset[0]
#print(f"[dataset]> {dataset} dataset graph...")
#print("\t->", graph)
#print("\t#nodes:", graph.num_nodes)
#print("\t#edges:", graph.num_edges)


## extract a random node to train on
l = range(400, 700, 5)
idx = torch.randint(0, len(l), (1,))
node_idx = torch.tensor([l[idx]]) 
print(f"\nChosing node {node_idx.item()}...")

edge_idx = graph.edge_index.indices()
#print("edge_idx:", edge_idx)
_, sub_index, _, _ = k_hop_subgraph(node_idx, 3, edge_idx)
print("\tnode neighborhood:", sub_index.size())

## node features
x = graph.x
print("\tnode features    :", x.size())


## instantiate GNNs model
model = NodeGCN(num_features=10, num_classes=4, device="cpu")
output = model(x, sub_index)[node_idx]
print("\nGNN output  :", output)

## need dense adjacency matrix for GCNSynthetic model
v = torch.ones(sub_index.size(1))
s = (graph.num_nodes,graph.num_nodes)
sub_index = torch.sparse_coo_tensor(indices=sub_index, values=v, size=s).to_dense()

model = GCNSynthetic(nfeat=10,nhid=20,nout=20,nclass=4,dropout=0.0)
output = model(x, sub_index)[node_idx]
print("CF-GNN output:", output)
