import os
import pickle as pkl

import torch
from torch_geometric.data import Data, InMemoryDataset

path_to_data = "/../../datasets/"
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_data


class BAGraphDataset(InMemoryDataset):
    r"""PyG dataset class to wrap the stored BA-Shapes datasets from the 
    `"GNNExplainer: Generating Explanations for Graph Neural Networks" 
    <https://arxiv.org/pdf/1903.03894.pdf>` paper.
    """
    def __init__(self, x, edge_index, labels, y_val, y_test, train_mask, val_mask, 
                test_mask, edge_label, transform=None, pre_transform=None, verbose: bool=False):
        """
        The parameters to initialize the class are the data loaded from the dataset stored in .pkl
        Args:
        - `x` : node features;
        - `edge_index`: adjacency matrix;
        - `labels`: ground-truth labels for node classification;
        - `y_val`:
        - `y_test`:
        - `train_mask`:
        - `val_mask`:
        - `test_mask`:
        - `edge_label`:
        """
        super().__init__(None, transform, pre_transform)

        x = torch.tensor(x, dtype=torch.float)
        num_nodes = x.size(0)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        expl_mask[torch.arange(400, num_nodes, 5)] = True

        # pyg uses sparse matrix representation as default
        edge_index = torch.tensor(edge_index).to_sparse()
        edge_label = torch.tensor(edge_label).to_sparse()

        labels[val_mask]  = y_val[val_mask]
        labels[test_mask] = y_test[test_mask]
        labels = torch.tensor(labels)
        if verbose:
            print("\n[DEBUG]> edge_index:", edge_index)
            print("\n[DEBUG]> edge_label:", edge_label)
            print("\n[DEBUG]> unique labels:", torch.unique(labels))

        data = Data(
            x=x, 
            edge_index=edge_index, 
            edge_label=edge_label,  
            y=labels, 
            train_mask=train_mask, 
            val_mask=val_mask, 
            test_mask=test_mask, 
            expl_mask=expl_mask)

        self.data, self.slices = self.collate([data])



def _load_node_dataset(dataset: str):
    """
    Load a graph dataset for graph node classification task.

    Args
    - `dataset`: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    
    Returns
        pytorch-geometric `Dataset`
    """
    filename = dataset + ".pkl"
    path = DATA_DIR + "pkls/" + filename
    print(f"\n[dataset]> node dataset from file '{filename}'...")

    # load raw data
    with open(path, 'rb') as fin:
        data = pkl.load(fin)
        adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = data

    # create dataset class with loaded data
    pyg_dataset = BAGraphDataset(
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
    print("\t#graphs:       ", len(pyg_dataset))
    print("\t#classes:      ", pyg_dataset.num_classes)
    print("\t#node_features:", pyg_dataset.num_node_features)

    return pyg_dataset

def load_dataset(dataset: str, paper: str="", skip_preproccessing: bool=False, shuffle: bool=True):
    """
    High level function which loads the dataset by calling the proper method 
    for node-classification or graph-classification datasets.

    Args:
    - `_dataset`: Which dataset to load. Choose from "syn1", "syn2", "syn3", 
        "syn4", "ba2" or "mutag";
    - `skip_preproccessing`: Whether or not to convert the adjacency matrix 
        to an edge matrix.
    - `shuffle`: Should the returned dataset be shuffled or not.
    
    Returns:    
        A couple (`BAGraphdataset()`,list). 
    """
    print(f"[dataset]> ...loading {dataset} dataset")
    if dataset[:3] == "syn": # Load node_dataset
        if dataset == "syn1" or dataset == "syn2":
            test_indices = range(400, 700, 5)
        elif dataset == "syn3":
            test_indices = range(511,871,6)
        elif dataset == "syn4":
            test_indices = range(511,800,1)

        return _load_node_dataset(dataset), test_indices
        
    #else: # Load graph dataset
    #    return load_graph_dataset(dataset, shuffle)