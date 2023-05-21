import os
import json
import pickle as pkl
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.data import Data, InMemoryDataset, Dataset

path_to_data = "/../../datasets/"
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_data


def parse_config(dataset: str, gnn: str):
    """Parse config file (.json) for chosen `dataset` and `gnn` into a dictionary."""
    if dataset == "syn1": data_cfg = dataset + "_BAshapes"
    elif dataset == "syn2": data_cfg = dataset + "_BAcommunities"
    elif dataset == "syn3": data_cfg = dataset + "_treeCycles"
    elif dataset in ["syn4","cfg_syn4"]: data_cfg = "syn4" + "_treeGrids"

    #gnn = "PGE" # to force PGE params
    rel_path = f"/../configs/{gnn}/{data_cfg}.json"
    cfg_path = os.path.dirname(os.path.realpath(__file__)) + rel_path

    try:    
        with open(cfg_path) as config_parser:
            config = json.loads(json.dumps(json.load(config_parser)))
        return config
    except FileNotFoundError:
        print(f"No config found for '{cfg_path}'")
        exit(0)

def get_neighborhood_labels(edge_index, edge_labels, test_indices):
    """ TODO: func to get labels for each node explanation separatedly.""" 
    return NotImplementedError("Nope, sorry.")

def syn_dataset_from_file(dataset: str, data_dir: str=DATA_DIR):
    """Loads data from a binary (.pkl) file representing one of the
    synthetic Barabasi-Albert graph datasets from the GNNExplainer paper.

    #### Args:

    dataset : (str)
        which synthetic dataset to load, one of "syn1", "syn2", "syn3", "syn4".
    
    data_dir : `str`
        directory path where dataset files are stored.
    """
    # syn4 (treeGrids) comes from CF-GNN, the others from PGE code
    f_type = ".pkl"
    filename = dataset + f_type
    path = data_dir + "pkls/" + filename
        
    # load raw data
    loaded_data = {}
    with open(path, 'rb') as fin:
        if dataset == "cfg_syn4":
            data = torch.load(fin)
            loaded_data = data
        else:
            data = pkl.load(fin)
            loaded_data["adj"]        = data[0]
            loaded_data["features"]   = data[1]
            loaded_data["y_train"]    = data[2]
            loaded_data["y_val"]      = data[3]
            loaded_data["y_test"]     = data[4]
            loaded_data["train_mask"] = data[5]
            loaded_data["val_mask"]   = data[6]
            loaded_data["test_mask"]  = data[7]
            loaded_data["edge_label"] = data[8]
    
    return loaded_data



class BAGraphDataset(Dataset):
    r"""PyG dataset class to wrap the synthetic BA-Shapes datasets from the 
    `"GNNExplainer: Generating Explanations for Graph Neural Networks"` 
    <https://arxiv.org/pdf/1903.03894.pdf> paper.
    """

    def __init__(self, 
            dataset  : str="syn1", 
            data_dir : str=DATA_DIR,
            load_adv : bool=False, 
            transform=None, 
            pre_transform=None, 
            verbose  : bool=False
        ):
        """The data are loaded from a stored `.pkl` file representing one of 
        the synthetic Barabasi-Albert graph datasets from the paper mentioned above. 
        
        #### Args:

        dataset : (str)
            which synthetic dataset to load, one of "syn1", "syn2", "syn3", "syn4".
        
        data_dir : `str`
            directory path where dataset files are stored. 
        """
        super().__init__(None, transform, pre_transform)   
        # Data are retrieved aws a dictonary with the following keys:
        # {adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label}
        data = syn_dataset_from_file(dataset=dataset,data_dir=data_dir)

        features = data["features"]
        x = torch.tensor(features, dtype=torch.float)
        num_nodes = x.size(0)
        expl_mask = torch.zeros(num_nodes, dtype=torch.bool)
        expl_mask[torch.arange(400, num_nodes, 5)] = True

        # pyg uses sparse matrix representation for adjacency matrix as default
        adj = data["adj"]
        edge_index = torch.tensor(adj).squeeze().to_sparse()

        edge_label_matrix = data["edge_label"]
        edge_label = torch.tensor(edge_label_matrix).to_sparse_coo()

        self.train_mask = data["train_mask"]
        self.val_mask   = data["val_mask"]
        self.test_mask  = data["test_mask"]

        y_train = data["y_train"]
        y_val = data["y_val"]
        y_test = data["y_test"]
        y_train[self.val_mask]  = y_val[self.val_mask]
        y_train[self.test_mask] = y_test[self.test_mask]
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
            expl_mask=expl_mask,
            n_id=torch.arange(num_nodes))

        # collate function needs a list of Data objects
        data_list = [data]

        # load raw adversarial examples
        if load_adv:
            adv_file = dataset + "_adv_train_CF-GNN.pt"
            adv_path = data_dir + "pkls/" + adv_file

            with open(adv_path, 'rb') as fin:
                adv_dict = torch.load(fin)
                adv_feats = adv_dict["node_feats"] #.clone().detach()

                if verbose:
                    print("[DEBUG]> adv_feat:", adv_feats.size())
                    for k,v in adv_dict.items():
                        print("[DEBUG]> key:",k,"item:",v)

            adv_data = Data(
                x=adv_feats,
                edge_index=edge_index.indices(), 
                edge_label=edge_label,  
                y=labels, 
                expl_mask=expl_mask)            

            data_list.append(adv_data)
        
        if verbose:
            for d in data_list:
                print("[DEBUG]> data_list:", d)

        #self.data, self.slices = self.collate(data_list)   # if using pyg InMemoryDataset
        self.data = data_list

    def len(self):
        """Returns the number of examples in your dataset."""
        return len(self.data)

    def get(self, idx):
        """Implements the logic to load a single graph."""
        #data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return self.data[idx]



def load_dataset(dataset: str, paper: str="", load_adv: bool=False, skip_preproccessing: bool=False, shuffle: bool=True):
    r"""High level function which loads the dataset by calling the proper method 
    for node-classification or graph-classification datasets.

    Args:
    - `_dataset`(str): Which dataset to load. Choose from "syn1", "syn2", "syn3", 
        "syn4", "ba2" or "mutag";
    - `load_adv`(bool): whether or not to load the adversarial example graph;
    - `skip_preproccessing`(bool): Whether or not to convert the adjacency matrix 
        to an edge matrix.
    - `shuffle`(bool): Should the returned dataset be shuffled or not.
    
    Returns:    
        A couple (`torch_geometric.data.Dataset`,list). 
    """
    print(Fore.GREEN + f"[dataset]> loading dataset...")
    if "syn" in dataset: 
        # Load node-classification datasets
        if dataset == "syn1" or dataset == "syn2":
            test_indices = range(400, 700, 5)
        elif dataset == "syn3":
            test_indices = range(511,871,6)
        elif dataset in ["syn4","cfg_syn4"]:
            test_indices = range(511,800,1)

        filename = dataset + ".pkl"
        print(Fore.GREEN + "[dataset]> node dataset from file",f"'{filename}'")

        # create dataset class with loaded data
        pyg_dataset = BAGraphDataset(dataset=dataset, load_adv=load_adv)
        print("\t>> #graphs:       ", len(pyg_dataset))
        print("\t>> #classes:      ", pyg_dataset.num_classes)
        print("\t>> #node_features:", pyg_dataset.num_node_features)

        return pyg_dataset, test_indices
        
    else: 
        # TODO Load graph-classification datasets
        #return load_graph_dataset(dataset, shuffle)
        return NotImplementedError("Graph classification datasets not yet implemented.")