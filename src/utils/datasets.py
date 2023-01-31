import os
import json
import pickle as pkl
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.data import Data, InMemoryDataset

path_to_data = "/../../datasets/"
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_data


def parse_config(config_path: str):
    """Parse config file (.json) at `config_path` into a dictionary"""
    try:    
        with open(config_path) as config_parser:
            config = json.loads(json.dumps(json.load(config_parser)))
        return config
    except FileNotFoundError:
        print("No config found")
        return None



class BAGraphDataset(InMemoryDataset):
    r"""PyG dataset class to wrap the synthetic BA-Shapes datasets from the 
    `"GNNExplainer: Generating Explanations for Graph Neural Networks"` 
    <https://arxiv.org/pdf/1903.03894.pdf> paper.
    """
    def __init__(self, dataset: str="syn1", transform=None, pre_transform=None, verbose: bool=False):
        r"""The data are loaded from a stored `.pkl` file representing one of 
        the synthetic Barabasi-Albert graph datasets from the paper mentioned above. 
        
        Args:
        - `dataset`(str): which synthetic dataset to load, one of "syn1", "syn2", "syn3", "syn4".
        """
        super().__init__(None, transform, pre_transform)
        
        filename = dataset + ".pkl"
        path = DATA_DIR + "pkls/" + filename
        # load raw data
        with open(path, 'rb') as fin:
            data = pkl.load(fin)
            adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = data

        self.train_mask = train_mask
        self.val_mask   = val_mask
        self.test_mask  = test_mask


        x = torch.tensor(features, dtype=torch.float)
        num_nodes = x.size(0)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        expl_mask[torch.arange(400, num_nodes, 5)] = True

        # pyg uses sparse matrix representation as default
        edge_index = torch.tensor(adj).to_sparse()
        edge_label = torch.tensor(edge_label_matrix).to_sparse()

        y_train[val_mask]  = y_val[val_mask]
        y_train[test_mask] = y_test[test_mask]
        labels = torch.tensor(y_train)
        if verbose:
            print("\n[DEBUG]> edge_index:", edge_index)
            print("\n[DEBUG]> edge_label:", edge_label)
            print("\n[DEBUG]> unique labels:", torch.unique(labels))

        data = Data(
            x=x, 
            edge_index=edge_index.indices(), 
            edge_label=edge_label, #.indices(),  
            y=labels, 
            expl_mask=expl_mask)

        self.data, self.slices = self.collate([data])


def _load_node_dataset(dataset: str):
    r"""Load a graph dataset for graph node classification task.

    Args
    - `dataset`: Which dataset to load. Choose from "syn1", "syn2", "syn3" or "syn4"
    
    Returns
        `torch_geometric.data.Dataset`
    """
    filename = dataset + ".pkl"
    print(Fore.GREEN + f"[dataset]> node dataset from file '{filename}'...")

    # create dataset class with loaded data
    pyg_dataset = BAGraphDataset(dataset=dataset)

    print("\t#graphs:       ", len(pyg_dataset))
    print("\t#classes:      ", pyg_dataset.num_classes)
    print("\t#node_features:", pyg_dataset.num_node_features)

    return pyg_dataset

def load_dataset(dataset: str, paper: str="", skip_preproccessing: bool=False, shuffle: bool=True):
    r"""High level function which loads the dataset by calling the proper method 
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
    print(Fore.GREEN + f"[dataset]> loading {dataset} dataset...")
    if dataset[:3] == "syn": 
        # Load node-classification datasets
        if dataset == "syn1" or dataset == "syn2":
            test_indices = range(400, 700, 5)
        elif dataset == "syn3":
            test_indices = range(511,871,6)
        elif dataset == "syn4":
            test_indices = range(511,800,1)

        return _load_node_dataset(dataset), test_indices
        
    else: 
        # TODO Load graph-classification datasets
        #return load_graph_dataset(dataset, shuffle)
        return NotImplementedError("Graph classification datasets not yet implemented.")