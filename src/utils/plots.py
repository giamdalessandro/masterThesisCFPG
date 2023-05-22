import networkx as nx
from matplotlib import pyplot as plt

from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



def plot_graph(edge_index, expl_weights, n_idx: int, e_cap: int=0, show: bool=True, verbose: bool=False):
    """Basic function to plot explanation subgraph."""
    if e_cap > 0:
        n_edges = e_cap
    elif e_cap == 0:
        over_thres = (expl_weights > 0.5).sum().item()
        over_mean = (expl_weights > torch.mean(expl_weights)).sum().item()
        n_edges = over_thres
        if verbose:
            print("\n\t[plot]> #over_mean :", over_mean)
            print("\t[plot]> #over_thres:", over_thres)

    # get most important edges from explanation weights
    sorted_edge_weigths, sorted_index = torch.sort(expl_weights, descending=True)
    if verbose: print("\t[plot]> sorted:", sorted_edge_weigths.size())
    sorted_edge_weigths = sorted_edge_weigths[:n_edges]

    # Initialize graph object
    G = nx.Graph()

    # extract edges
    sorted_egde_indices = sorted_index[:n_edges]
    edge_index = edge_index[:,sorted_egde_indices] 
    edges = [(e[0].item(),e[1].item()) for e in edge_index.T]
    #if verbose: print("\t[plot]> real edges:", len(edges))

    # add node and edges to final graph
    G.add_edges_from(edges)
    if verbose: 
        print("\t[plot]> G edges:", len(G.edges))
        print("\t[plot]> G nodes:", len(G.nodes))

    #G.add_nodes_from(nodes)

    pos = nx.spring_layout(G)
    if show: 
        nx.draw(G, pos=pos, with_labels=True, font_size=8)  # draw grpah
        try:
            nx.draw_networkx_nodes(G, pos, nodelist=[n_idx], node_color="orange")  # highlight explained node
        except nx.exception.NetworkXError as e:
            print(Fore.RED + f"[DEBUG]> {e}")
            
        plt.title(f"Node {n_idx} expl")
        plt.show()

def plot_expl_loss(expl_name: str, losses: dict, cf_num: list, cf_tot: int, show: bool=True):
    """Plot explainer training performance"""
    # normalize losses
    losses_l = [torch.Tensor(l) for l in losses.values()]
    losses_t = torch.stack(losses_l, dim=0).cpu()
    l_min = torch.min(losses_t, dim=1).values
    l_max = torch.max(losses_t, dim=1).values
    
    # normalize losses contribution for a better plot
    norm_losses = []
    for i in range(len(losses.keys())):
        norm_losses.append((losses_t[i] - l_min[i].item())/(l_max[i].item() - l_min[i].item()))

    x = range(1,len(cf_num)+1)   # epochs id
    plt.figure(figsize=(3, 4))

    # loss plot
    plt.subplot(211)
    plt.title("explanation loss")
    plt.plot(x, (norm_losses[0]).tolist(), ".-", label="loss")
    plt.plot(x, (norm_losses[1]).tolist(), "--", label="size loss")
    plt.plot(x, (norm_losses[2]).tolist(), "--", label="ent loss")
    plt.plot(x, (norm_losses[3]).tolist(), "--", label="pred loss")
    plt.xticks(x, minor=True)
    plt.ylabel("normalized loss scores")
    plt.grid(which="major", alpha=0.5)
    plt.grid(which="minor", alpha=0.2)
    plt.legend()

    # cf examples plot
    plt.subplot(212)
    plt.title("cf examples found")
    plt.plot(x, cf_num, ".-", color="magenta")
    plt.xlabel("epoch")
    plt.ylabel("#cf examples")
    plt.xticks(x, minor=True)
    plt.grid(which="major", axis="x", alpha=0.5)
    plt.grid(which="minor", axis="x", alpha=0.2)
    
    # cf perc twinx plot
    cf_perc = sorted(list(set([f"{(f/cf_tot):.4f}" for f in cf_num])))
    perc_tick = sorted(list(set(cf_num)))
    print("\t>> perc:", cf_perc)
    print("\t>> perc:", perc_tick)

    plt.twinx()
    plt.ylabel("portion of cf found")
    plt.plot(x, cf_num, alpha=0.1)
    plt.yticks(ticks=perc_tick,labels=cf_perc)
    plt.grid(which="major", axis="y", alpha=0.8, color="gray")
    #plt.hlines(perc_tick, 1, len(x), "gray", "--", alpha=0.2)

    plt.suptitle(f"{expl_name} training")

    if show:
        plt.show()

    return