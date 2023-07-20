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
    
    print("\n[plots]> Generating --roc plots...")
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
    print("\t>>[ax1]> explanation loss plot... OK")


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
    print("\t>>[ax2]> loss contribution plot... OK")

    ### cf examples plot
    if len(cf_num) >= 0 and cf_tot > 0:  # cf examples plot
        perc_tick = sorted(list(set(cf_num)))
        #y_ticks = [0] + perc_tick
        y_ticks = list(range(0,cf_tot,10)) if cf_tot <= 60 else list(range(0,cf_tot,30))
        y_ticks = y_ticks if y_ticks[-1] == cf_tot else y_ticks + [cf_tot]
        
        ax3.set_title("cf examples found (training)")
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
        print("\t\t>> cf_perc:",len(cf_perc))
        print("\t\t>> cf_num :",len(cf_num))

        #cf_mid = cf_num[(len(cf_num)//2)-1]
        #cf_num_tx = [min(cf_num),cf_mid,max(cf_num)] if min(cf_num) != max(cf_num) else cf_num[-1]
        #cf_perc_mid = cf_perc[(len(cf_perc)//2)-1] if cf_perc[-1] == "1.0000" else cf_perc[(len(cf_perc)//2)-2] 
        #cf_ticks = [min(cf_perc)[:4],cf_perc_mid[:4],max(cf_perc)[:4]] if min(cf_perc) != max(cf_perc) else cf_perc[-1][:4]
        #if cf_ticks[-1] != "1.00":
        #    cf_ticks = cf_ticks + ["max"]
        #    cf_num_tx = cf_num_tx + [cf_tot]

        ax3_tx = ax3.twinx()
        ax3_tx.set_ylabel("portion of cf found")
        ax3_tx.plot(x, cf_num, ".-", alpha=0.2)
        ax3_tx.set_yticks(y_ticks, minor=True)
        #ax3_tx.set_yticks(ticks=cf_num_tx, labels=cf_ticks)
        ax3_tx.grid(which="major", axis="y", alpha=0.4, color="gray")
        #ax3_tx.legend()
        #plt.hlines(perc_tick, 1, len(x), "gray", "--", alpha=0.2)
        print("\t>>[ax3]> cf examples plot... OK")

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
    print("\t>>[ax4]> ROC curve plot... OK")


    fig.suptitle(f"{expl_name} training on {dataset.upper()} dataset")
    if show:   plt.show()
    return


def plot_scatter_node_mask(explanations: list, show: bool=True):
    """Plot density of edge weights produced by the explainer.
    - explanations is a list of tuples (edge-idx,weights,node-idx)"""
    val_dict = {"check":[]}
    for i in range(10):
        val_dict[str(i)] = []

    # get explanation mask value ranges 
    print("\n[plot]> counting elems in span...")
    tot_edges = 0
    #explanations = explanations[:6]  # to fix plot, show only six nodes expl.
    for expl in explanations:
        _, e, _ = expl
        e = e.detach()
        n_edges = e.size(0)
        tot_edges += n_edges

        val_dict["0"].append(e[(e >= 0.0) & (e < 0.1)])
        val_dict["1"].append(e[(e >= 0.1) & (e < 0.2)])
        val_dict["2"].append(e[(e >= 0.2) & (e < 0.3)])
        val_dict["3"].append(e[(e >= 0.3) & (e < 0.4)])
        val_dict["4"].append(e[(e >= 0.4) & (e < 0.5)])
        val_dict["5"].append(e[(e >= 0.5) & (e < 0.6)])
        val_dict["6"].append(e[(e >= 0.6) & (e < 0.7)])
        val_dict["7"].append(e[(e >= 0.7) & (e < 0.8)])
        val_dict["8"].append(e[(e >= 0.8) & (e < 0.9)])
        val_dict["9"].append(e[(e >= 0.9) & (e < 1.0)])
        val_dict["check"].extend(e[(e >= 1.0)])

    ## log to terminal global count and percentages 
    #val_count = []   # global count for each interval
    #val_perc  = []   # global perc for each interval
    #for k,v in val_dict.items():
    #    k_cnt = 0 
    #    for e in range(len(v)):
    #        k_cnt += len(v[e])
    #    val_count.append(k_cnt)
    #    val_perc.append(k_cnt/tot_edges)
    #    if k == "check": print(f"\t>> {k} -> {k_cnt/tot_edges:.2f} %, {k_cnt} elems")
    #    else: print(f"\t>> {k}-{int(k)+1}\t-> {k_cnt/tot_edges:.2f} %, {k_cnt} elems")

    plt.style.use("bmh") # seaborn-v0_8
    fig = plt.figure(figsize=(9,9), layout="tight")
    ax1 = fig.add_subplot(1,1,1)   # hist 1 

    ## expl-wise count and percentages
    x = [round(x/10,1) for x in range(0,10,1)]
    all_values = []
    for e in range(len(explanations)):
        n_id = explanations[e][2]
        node_v_cnt = []
        tot_v_cnt = 0
        for i in range(10):
            tot_v_cnt += len(val_dict[str(i)][e])
            node_v_cnt.append(len(val_dict[str(i)][e]))
            all_values.extend(val_dict[str(i)][e])

        perc_v_cnt = [(c/tot_v_cnt) for c in node_v_cnt]
        ax1.scatter(x, perc_v_cnt, alpha=0.5, s=node_v_cnt, label=str(n_id))

    x_ticks = x
    x_labels = [f"{i}-{round(i+0.1,1)}" for i in x_ticks[:-1]] + ["0.9-1"]
    y_ticks = [float(f"0.{yt}") for yt in range(0,10,2)] + [1.0]
    y_ticks_minor = [float(f"0.{yt}") for yt in range(0,10,1)] + [1.0]
    y_labels = [f"{yl}%" for yl in range(0,100,20)] + ["100%"]

    # [ax3] node-wise percentage scatter plot
    ax1.set_title("Per-node mask value distribution")
    ax1.set_ylabel("% of values (per-node) in range")
    ax1.set_yticks(ticks=y_ticks, labels=y_labels)
    ax1.set_yticks(ticks=y_ticks_minor, minor=True)
    ax1.set_xticks(ticks=x_ticks, labels=x_labels, rotation=-45)
    ax1.set_xlim(-0.05,0.95)
    ax1.grid(alpha=0.4, which="both")
    #ax3.legend(markerscale=0.5)


    fig.suptitle(f"Explanation mask (per-node) values distribution")
    if show:   plt.show()
    return

def plot_mask_density(explanations: list, em_logs: dict, dataset: str, epochs: int, show: bool=True):
    """Plot density of edge weights produced by the explainer.
    - explanations is a list of tuples (edge-idx,weights,node-idx)"""

    # get explanation mask value ranges 
    print("\n[plot]> counting elems in span...")
    tot_edges = 0
    all_values = []
    nodes = [] 
    over_mean = []
    #explanations = explanations[:6]  # to fix plot, show only six nodes expl.
    for expl in explanations:
        edge_idx, e, n_idx = expl
        e = e.detach()
        tot_edges += e.size(0)
        all_values.extend(e)

        nodes.append(str(n_idx))
        m, std = torch.std_mean(e, unbiased=False)
        thres = m + std
        over_mean.extend([n_idx for _ in range((e > thres).long().sum().item())])

    ### TODO potrei fare uno scatter per ogni nodo ed i valori della sua explanation mask
    # - posso aggiungere un histogramma del mappazzone fnale che ne viene fuori
    plt.style.use("bmh") # seaborn-v0_8

    fig = plt.figure(figsize=(9,9), layout="tight")
    ax1 = fig.add_subplot(2,1,1)  # hist 1 
    #ax2 = fig.add_subplot(3,1,2)  # hist 2 
    ax3 = fig.add_subplot(2,1,2)  # hist 3 / bar plot 3  

    ### expl-wise count and percentages
    x = [round(x/10,1) for x in range(0,10,1)]
    x_ticks = x

    ### [ax1] global bin.concrete hist plot
    ax1.set_title("post-sampling (expl.out)")
    _, _, bars = ax1.hist(all_values, 
                        bins=20, 
                        range=(0,1), 
                        align="mid", 
                        edgecolor="white",
                        density=False)
    ax1.bar_label(bars)
    #ax1.set_xlabel("value ranges")
    ax1.set_ylabel("no. values in range")
    ax1.set_xticks(ticks=x_ticks + [1.0])
    #ax1.set_yticks(ticks=y_ticks, labels=y_labels)
    ax1.set_xlim(-0.05,1.05)
    ax1.grid(alpha=0.4)

    """
    ### [ax2] global pre-sample hist plot
    print("\t>> pre-sample:", len(em_logs["pre-sample"]))
    all_pre = []
    all_gcn = []
    for l in range(len(em_logs["pre-sample"])):
        all_pre.extend(em_logs["pre-sample"][l].squeeze())
        all_gcn.extend(em_logs["post-gcn"][l].squeeze())

    max2 = max(all_pre).item()
    min2 = min(all_pre).item()    
    bin_size = round((max2-min2)/20, 2)
    print("\t>> bin_size:", bin_size)

    ax2.set_title("pre-sampling (mlp out)")
    _, _, bars = ax2.hist(all_pre, 
                        bins=20,
                        align="mid", 
                        edgecolor="white")
                        #density=False)
    ax2.bar_label(bars)
    ax2.set_xlabel("value ranges")
    ax2.set_ylabel("no. values in range")
    ax2.set_xlim((min2-bin_size),(max2+bin_size))
    ax2.grid(alpha=0.4)


    ### [ax3] global pre-sample hist plot
    x3 = list(range(0,len(nodes)))
    width = 1
    print("\t>> [ax3]> nodes:", len(nodes))
    print("\t>> [ax3]> over mean:", len(over_mean))
    print("\t>> [ax3]> x3:", len(x3))"""

    ax3.set_title("Mask no. edges post-sampling (over-mean)")
    #ax3.bar(x3, over_mean, width=width, label="no expl.edges")
    _, _, bars = ax3.hist(over_mean, 
                        bins=len(nodes), 
                        align="left", 
                        edgecolor="white")
                        #range=(0,1), 
                        #density=False)
    #ax3.bar_label(bars)
    ax3.set_xlim(int(nodes[0])-5,int(nodes[-1])+5)
    ax3.set_xlabel("node id")
    ax3.set_ylabel("no. edges")
    ax3.grid(alpha=0.4)

    #ax3.set_title("Expl. mask value distribution")
    #_, _, bars = ax3.hist(em_logs["post-gcn"]) #, 
                        #bins=10, 
                        #range=(0,1), 
                        #align="mid", 
                        #edgecolor="white",
                        #density=False)
    #ax3.bar_label(bars)


    fig.suptitle(f"Explanation mask values distribution - {dataset.upper()}, {epochs} epochs")
    if show:   plt.show()
    return

