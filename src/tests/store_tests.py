import os
import torch 
import pickle as pkl
from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

path_to_data = "/../../datasets/"
DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + path_to_data



def load_saved_test(path: str):
    rel_path = path_to_data + path
    save_path = os.path.dirname(os.path.realpath(__file__)) + rel_path

    # load raw data
    with open(save_path, 'rb') as fin:
        #try:
        #    data = torch.load(fin)
        #except:
        data = pkl.load(fin)    

        print(">> reading from",path)
        print(">>", len(data))

        #adj = torch.Tensor(data["adj"]).squeeze().to_sparse_coo()
        #feats = torch.Tensor(data["features"])
        adj = torch.Tensor(data[0]).squeeze().to_sparse_coo()
        feats = torch.Tensor(data[1])

        print(">> adj  :", adj.indices().size())
        print(">> feats:", feats.size())


def syn_dataset_from_file(dataset: str, data_dir: str=DATA_DIR, save: bool=False):
    """Loads data from a binary (.pkl) file representing one of the
    synthetic Barabasi-Albert graph datasets from the GNNExplainer paper.

    #### Args:

    dataset : (str)
        which synthetic dataset to load, one of "syn1", "syn2", "syn3", "syn4".
    
    data_dir : `str`
        directory path where dataset files are stored.
    """
    # syn4 (treeGrids) comes from CF-GNN (.pickle), the others from PGE code (.pkl)
    f_type = ".pickle" #if dataset == "syn4" else ".pkl"
    filename = dataset + f_type
    path = data_dir + "pkls/" + filename

    #filename = dataset + ".pkl"
    #path = data_dir + "pkls/" + filename
    ## load raw data
    #with open(path, 'rb') as fin:
    #    data = pkl.load(fin)
    #    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, edge_label_matrix = data

    # load raw data
    loaded_d = {}
    with open(path, 'rb') as fin:
        print(f">> loading from {path} ...")
        data = pkl.load(fin)
        print(f">> elems: {len(data)}")

        loaded_d["adj"]        = data["adj"]
        loaded_d["features"]   = data["feat"]
        loaded_d["labels"]     = data["labels"]
        loaded_d["train_mask"] = data["train_idx"]
        loaded_d["test_mask"]  = data["test_idx"]

    print(f">> new dict: {loaded_d.keys()}")
    for k,v in loaded_d.items():
        print(f"\t{k} \t- {torch.Tensor(v).size()}")


    loaded_old = {}
    path = data_dir + "pkls/" + "syn4_old.pkl"
    with open(path, 'rb') as fin:
        print(f"\n>> loading from {path} ...")
        data = pkl.load(fin)
        print(f">> elems (old): {len(data)}")

        loaded_old["adj"]        = data[0]
        loaded_old["features"]   = data[1]
        loaded_old["y_train"]    = data[2]
        loaded_old["y_val"]      = data[3]
        loaded_old["y_test"]     = data[4]
        loaded_old["train_mask"] = data[5]
        loaded_old["val_mask"]   = data[6]
        loaded_old["test_mask"]  = data[7]
        loaded_old["edge_label"] = data[8]

    print(f">> old dict: {loaded_old.keys()}")
    for k,v in loaded_old.items():
        print(f"\t{k} \t- {torch.Tensor(v).size()}")

    merged = {}
    for k,v in loaded_old.items():
        merged[k] = v

    merged["adj"] = loaded_d["adj"]

    if save: torch.save(merged, "./src/tests/test_syn5-4.pkl")
    return


#def create_test_dataset():
#    """Create"""
#    for i in range(4):
#        dataset = f"syn{i}.pkl"



if __name__ == "__main__":
    #syn_dataset_from_file(dataset="syn5", save=False)
    load_saved_test("pkls/pge_syn4.pkl")
