import os
import pickle as pkl

import torch
from torch_geometric.data import Data, InMemoryDataset

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + "/../../datasets/"


class MyBAshapes(InMemoryDataset):
    r"""Class to wrap the stored BA-Shapes dataset from the `"GNNExplainer: Generating Explanations
    for Graph Neural Networks" <https://arxiv.org/pdf/1903.03894.pdf>` paper.

    Args:
    - `x`: node features;
    - `edge_index`: adjacency matrix;
    - `labels`: ground-truth labels for node classification;
    - `y_val`:
    - `y_test`:
    - `train_mask`:
    - `val_mask`:
    - `test_mask`:
    - `edge_label`:
    """
    def __init__(self, x, edge_index, labels, y_val, y_test, train_mask, val_mask, 
                test_mask, edge_label, transform=None, pre_transform=None, verbose: bool=False):
        super().__init__(None, transform, pre_transform)

        x = torch.tensor(x, dtype=torch.float)
        num_nodes = x.size(0)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        expl_mask[torch.arange(400, num_nodes, 5)] = True

        edge_index = torch.tensor(edge_index).to_sparse()
        edge_label = torch.tensor(edge_label).to_sparse()

        labels[val_mask]  = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]
        labels = torch.tensor(labels)
        if verbose:
            print("\n[DEBUG]> edge_index:", edge_index)
            print("\n[DEBUG]> edge_label:", edge_label)
            print("\n[DEBUG]> unique labels:", torch.unique(labels))


        data = Data(x=x, edge_index=edge_index, edge_label=edge_label, y=labels,
                    train_mask=train_mask, val_mask=val_mask, test_mask=test_mask, expl_mask=expl_mask)

        self.data, self.slices = self.collate([data])



def load_node_dataset(dataset: str):
    """
    Load a node dataset.

    Args
    - `dataset`: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    
    Returns
        pytorch-geometric `Dataset`
    """
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dataset + ".pkl"
    path = DATA_DIR + "pkls/" + filename
    print(f"\n[dataset]> loading dataset from '{filename}'...")

    with open(path, 'rb') as fin:
        data = pkl.load(fin)
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = data

    # create dataset class with loaded data
    pyg_dataset = MyBAshapes(
        x=features, 
        edge_index=adj, 
        labels=y_train,
        y_val=y_val,
        y_test=y_test,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        edge_label=edge_label_matrix
    )
    print("\t#entries:      ", len(pyg_dataset))
    print("\t#classes:      ", pyg_dataset.num_classes)
    print("\t#node_features:", pyg_dataset.num_node_features)

    return pyg_dataset