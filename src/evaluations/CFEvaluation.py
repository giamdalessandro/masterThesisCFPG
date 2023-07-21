import torch
from tqdm import tqdm

from colorama import init, Fore 
init(autoreset=True) # initializes Colorama


def sparsity_score(explanations):
    """Compute sparsity score i.e. the proportion of edges that are removed
    ([Yuan et al., 2020](https://arxiv.org/abs/2012.15445))"""
    node_spa = []
    for expl in explanations:
            edge_idx, e, n_idx = expl
            n_edges = edge_idx.size(1).item()
            removed = (e > e.mean()).long().sum().item()

            node_spa.append(1 - round(removed/n_edges,4))  

    avg_spa = torch.FloatTensor(node_spa).mean().item()
    return avg_spa.item()

def get_cf_metrics(edge_labels: str, explanations: list, counterfactuals: dict, n_nodes: int, thres: float, task: str="node"):
    """Computes common metrics for a Countefactual Explainers. 
    ([Lucic et al., 2022](https://arxiv.org/abs/2102.03322))
    
    - Fidelity: proportion of nodes where the original predictions matches 
        the prediction for the explanations; 
    - Sparsity: the proportion of edges that are removed; 
    - Accuracy: mean proportion of explanations that are “correct” (i.e. 
        part of the explanation ground truth)
    - Explanation Size: the number of removed edges;
    
    #### Args
    ground_truth : `list` 
        list of edges ground-truth
    
    explanations : `list` 
        list of produxed explanations
    
    counterfactuals: `dict`
        dict containing the CF examples found, where key is the node-id
        and value is the CF example.
    
    task : `str`
        Whether to compute metrics for node or graph classification.

    #### Retruns 
        A tuple with the scores for (Fidelity, Sparsity, Expl.Size, Accuracy).
    """
    print(Fore.MAGENTA + "\n[metrics]> computing CF metrics...")
    if task == "graph":
        return NotImplementedError("Graph classification not yet implemented.")
    elif task == "node":
        # loop over the explanation to compute sparsity and expl.size metrics
        edge_labels = edge_labels["per_node_labels"]
        node_spa = []
        node_acc = []
        expl_size = []
        for expl in (t := tqdm(explanations, desc="[metrics]> (Fid,Spa,Acc,Size)", colour="magenta")):
                edge_idx, e_mask, n_idx = expl

                #m, std = torch.std_mean(e_mask, unbiased=False)
                #thres = m + std
                e_mask = (e_mask > thres).detach().long() #e_mask.mean()

                n_edges = edge_idx.size(1)
                n_removed = e_mask.sum().item()
                node_spa.append(1 - round(n_removed/n_edges,4)) 
                expl_size.append(n_removed) 

                dense = torch.zeros((n_nodes,n_nodes)).long()
                dense[edge_idx[0],edge_idx[1]] = e_mask
                motif_gt = torch.LongTensor(edge_labels[str(n_idx)]).T

                if n_removed == 0.0: n_removed = 0.1
                in_expl = dense[motif_gt[0],motif_gt[1]].sum().item()
                node_acc.append(round(in_expl/n_removed,4))

        # compute sparsity, accuracy, explanation size
        spa = torch.FloatTensor(node_spa).mean().item()    # avg. sparsity score
        acc = torch.FloatTensor(node_acc).mean().item()    # avg. accuracy score
        size = torch.FloatTensor(expl_size).mean().item()  # avg. explanation size

        # compute fidelity score
        fnd = len(counterfactuals.keys())
        max_cf = len(explanations)
        fid = 1 - round(fnd/max_cf,4)                      # avg. fidelity score

        return fid, spa, acc, size
