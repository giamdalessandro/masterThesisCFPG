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

def plot_expl_loss(
        expl_name: str, 
        dataset: str,
        losses: dict, 
        cf_num: list, 
        cf_tot: int, 
        roc_gt: list, 
        roc_preds: list, 
        show: bool=True
    ):
    """Plot explainer training performances. The visual includes:
    - the explainer loss and its components (pred, size and ent losses);
    - the counterfactual examples found by the explainer;
    - the ROC curve plot (AUC score). 
    """
    # normalize losses contribution for a better plot
    losses_l = [torch.Tensor(l) for l in losses.values()]
    losses_t = torch.stack(losses_l, dim=0).cpu()
    l_min = torch.min(losses_t, dim=1).values
    l_max = torch.max(losses_t, dim=1).values
    
    # min-max normalization: (x - x_min)/(x_max - x_min)
    norm_losses = []
    for i in range(len(losses.keys())):
        norm_losses.append((losses_t[i] - l_min[i].item())/(l_max[i].item() - l_min[i].item()))

    # plot x ticks
    x_maj = range(0,len(norm_losses[0])+1,5) 
    x_maj = [1] + list(x_maj[1:]) 
    x = range(1,len(norm_losses[0])+1)   # epochs id
    
    fig = plt.figure(figsize=(12,9), layout="tight")
    ax1 = fig.add_subplot(3,5,(1,3))
    ax2 = fig.add_subplot(3,5,(6,8))
    ax3 = fig.add_subplot(3,5,(11,13))
    ax4 = fig.add_subplot(3,5,(4,10))


    ### losses plot
    # TODO: should enforce same colors on same loss components
    ax1.set_title("explanation loss")
    ax1.plot(x, (norm_losses[1]).tolist(), ".--", label="size loss", alpha=0.5)
    ax1.plot(x, (norm_losses[2]).tolist(), ".--", label="ent loss" , alpha=0.5)
    ax1.plot(x, (norm_losses[3]).tolist(), ".--", label="pred loss", alpha=0.5)
    ax1.plot(x, (norm_losses[0]).tolist(), ".-", label="loss", color="k")
    ax1.set_xticks(x_maj)
    ax1.set_xticks(x, minor=True)
    ax1.set_ylabel("normalized loss score")
    ax1.grid(which="major", alpha=0.5)
    ax1.grid(which="minor", alpha=0.2)
    ax1.legend()


    ### perc losses plot
    width = 0.25
    perc_losses = []
    for i in range(1,len(losses.keys())):
        perc_losses.append(torch.div(losses_t[i],losses_t[0].abs()))

    bottom_1 = perc_losses[0]
    bottom_2 = perc_losses[0] + perc_losses[1]
    #plt.bar(x, (perc_losses[0]).tolist(), label="loss")
    ax2.set_title("losses contribution")
    ax2.set_xticks(x_maj)
    ax2.set_xticks(x, minor=True)
    ax2.bar([t+(width*-1) for t in x], (perc_losses[0]).tolist(), width=width, label="size loss")
    ax2.bar([t+(width*0) for t in x], (perc_losses[1]).tolist(), width=width, label="ent loss" )
    ax2.bar([t+(width*1) for t in x], (perc_losses[2]).tolist(), width=width, label="pred loss")
    ax2.grid(which="major", alpha=0.5)
    ax2.grid(which="minor", alpha=0.2)
    ax2.legend(ncols=3)


    ### cf examples plot
    if len(cf_num) >= 0 and cf_tot > 0:  # cf examples plot
        perc_tick = sorted(list(set(cf_num)))
        #y_ticks = [0] + perc_tick
        y_ticks = list(range(0,cf_tot,10)) if cf_tot <= 60 else list(range(0,cf_tot,30))
        y_ticks = y_ticks if y_ticks[-1] == cf_tot else y_ticks + [cf_tot]
        
        ax3.set_title("cf examples found")
        ax3.plot(x, cf_num, "-", drawstyle='steps-mid', color="magenta", label="cf ex. found")
        for i in range(len(x)):
            ax3.text(x[i]-0.1, cf_num[i]+0.5, str(cf_num[i]), fontsize=9)
        ax3.set_xlabel("epoch")
        ax3.set_ylabel("no. cf examples")
        ax3.set_xticks(x_maj)
        ax3.set_xticks(x, minor=True)
        ax3.set_yticks(y_ticks)
        ax3.grid(which="major", axis="x", alpha=0.5)
        ax3.grid(which="minor", axis="x", alpha=0.2)
        ax3.legend(loc='upper left')
        
        ### cf perc twinx plot
        cf_perc = sorted(list(set([f"{(f/cf_tot):.4f}" for f in cf_num])))
        cf_mid = cf_num[(len(cf_num)//2)-1]
        cf_num_tx = [min(cf_num),cf_mid,max(cf_num)] if min(cf_num) != max(cf_num) else cf_num[-1]
        cf_perc_mid = cf_perc[(len(cf_num)//2)-1] if cf_perc[-1] == "1.0000" else cf_perc[(len(cf_num)//2)-2] 
        cf_ticks = [min(cf_perc)[:4],cf_perc_mid[:4],max(cf_perc)[:4]] if min(cf_perc) != max(cf_perc) else cf_perc[-1][:4]
        if cf_ticks[-1] != "1.00":
            cf_ticks = cf_ticks + ["max"]
            cf_num_tx = cf_num_tx + [cf_tot]

        ax3_tx = ax3.twinx()
        ax3_tx.set_ylabel("portion of cf found")
        ax3_tx.plot(x, cf_num, ".-", alpha=0.2)
        ax3_tx.set_yticks(y_ticks, minor=True)
        ax3_tx.set_yticks(ticks=cf_num_tx, labels=cf_ticks)
        ax3_tx.grid(which="major", axis="y", alpha=0.4, color="gray")
        #ax3_tx.legend()
        #plt.hlines(perc_tick, 1, len(x), "gray", "--", alpha=0.2)

    ### ROC curve plot
    from sklearn.metrics import RocCurveDisplay

    RocCurveDisplay.from_predictions(
        roc_gt,
        roc_preds,
        name=f"expl.edges vs the rest",
        color="darkorange",
        ax=ax4
    )
    ax4.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
    ax4.axis("square")
    ax4.set_xlabel("FP rate")
    ax4.set_ylabel("TP rate")
    ax4.set_title("ROC curve")
    ax4.legend()


    fig.suptitle(f"{expl_name} training on {dataset.upper()} dataset")
    if show:   plt.show()
    return