import networkx as nx
from matplotlib import pyplot as plt
#from matplotlib import pylab as pl

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx



def plot_graph(edge_index, expl_weights, n_idx: int, show: bool=True, verbose: bool=False):
    """Basic function to plot a graph."""
    n_edges = 12

    # get most important edges from explanation weights
    sorted_edge_weigths, sorted_index = torch.sort(expl_weights, descending=True)
    sorted_edge_weigths = sorted_edge_weigths[:n_edges]
    if verbose: print("\n[plot]> sorted:", sorted_edge_weigths)

    # Initialize graph object
    G = nx.Graph()

    # extract edges
    sorted_egde_indices = sorted_index[:n_edges]
    edge_index = edge_index[:,sorted_egde_indices] 
    edges = [(e[0].item(),e[1].item()) for e in edge_index.T]
    if verbose: print("[plot]> real edges:", edges)

    # add node and edges to final graph
    G.add_edges_from(edges)
    if verbose: 
        print("[plot]> G edges:", G.edges)
        print("[plot]> G nodes:", G.nodes)

    #G.add_nodes_from(nodes)

    pos = nx.spring_layout(G)
    if show: 
        nx.draw(G, pos=pos, with_labels=True, font_size=8)                     # draw grpah
        try:
            nx.draw_networkx_nodes(G, pos, nodelist=[n_idx], node_color="orange")  # highlight explained node
        except nx.exception.NetworkXError as e:
            print("[DEBUG]>", e)
            
        plt.title(f"Node {n_idx} expl")
        plt.show()