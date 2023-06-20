import os
import torch 
import json
import pickle as pkl
import pandas as pd
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


def store_run_csv(save_path: str=""):
    """Create dafafadxca"""
    
    data = {"cacca" : [3], "pupu": [5]}
    df = pd.DataFrame.from_dict(data)

    print(df)

    return


def clean_sep_lables_data(dataset: str="syn1"):
    full_path = DATA_DIR + f"/{dataset}_sep_labels.json"

    motif_size = 6 if dataset != "syn4" else 12
    to_write = {
        "dataset" : dataset,
        "motif_size" : motif_size,
        "n_nodes" : 0,
        "per_node_labels" : {}
    }

    with open(full_path, "r") as fr:
        data = json.load(fr)
        print(f"\n[{dataset}]> data loaded...")
        print(">> num of labeled:", len(data.keys()))
        #print(">> labeled nodes :", list(data.keys()))

        for k,v in data.items():
            if v["n_edges"] > motif_size:
                correct = []
                for i,j in v["expl_edges"]:
                    if (i >= int(k))  and (j >= int(k)): correct.append([i,j])

                to_write["per_node_labels"][k] = str(correct)

            else:
                to_write["per_node_labels"][k] = str(v["expl_edges"])

            #if len(to_write["per_node_labels"][k]) != motif_size:
            #    exit(f"[ERR]> {k} has {len(to_write['per_node_labels'][k])}... Daina cinciallegra!!!")

    to_write["n_nodes"] = len(to_write["per_node_labels"].keys())
    print("\nlabeled found:", len(to_write["per_node_labels"].keys()))
    
    with open(full_path[:-5] + "_clean.json", "w+") as fw:
        json.dump(to_write,fw,indent=4)
        fw.close()

    ### health check on labels (explain.py)
    #print("\n\t>> expl labels matrix:", explainer.correct_labels.size())
    #print("\t>> correct expl labels :", explainer.correct_labels.sum())
    #print("\t>> original expl labels:", dataset.get(0).edge_label.values().sum())
    #
    #fuffa = True
    #if fuffa:
    #    import json
    #
    #    thres = 10 if DATASET == "syn4" else 6
    #    to_json = {}
    #    for k,v in explainer.labeled_nodes.items():
    #        #print(f"\t>> node {k}, edges -> {v['n_edges']}")
    #        if v['n_edges'] >= thres: 
    #            to_json[k] = v
    #    print(f"\n\t>> found labels for {len(explainer.labeled_nodes.keys())} nodes")
    #
    #    with open(f"./datasets/{DATASET}_sep_labels.json","w+") as f:
    #        json.dump(to_json, f, indent=4)
    #        f.close()

    return



if __name__ == "__main__":
    #syn_dataset_from_file(dataset="syn5", save=False)
    #load_saved_test("pkls/pge_syn4.pkl")
    #store_run_csv()
    clean_sep_lables_data("syn4")