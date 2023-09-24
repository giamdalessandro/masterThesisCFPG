from colorama import init, Fore 
init(autoreset=True) # initializes Colorama

from explainers.PGExplainer import PGExplainer
from explainers.CFPGExplainer import CFPGExplainer
from explainers.CFPGv2 import CFPGv2
from explainers.OneHopExplainer import OneHopExplainer, PerfectExplainer
from explainers.CFGNNExplainer import CFGNNExplainer
#from utils.graphs import normalize_adj


def explainer_selector(cfg, model, graph, s_args, verbose: bool=False):
    """Select explainer model for the given params"""
    expl = s_args.explainer
    epochs = s_args.epochs
    device = s_args.device
    if verbose: print(Fore.MAGENTA+"\n[explain]> Loading",f"{expl}",Fore.MAGENTA+"explainer..")

    # if params are not given in the command line, use values from config file
    args_d = vars(s_args)
    on_file = []
    for k,v in cfg["expl_params"].items():
            try:
                if args_d[k] != -1.0 and args_d[k] != "base":
                    cfg["expl_params"][k] = args_d[k]
            except KeyError:
                on_file.append(k)

    #if verbose: print(f"\t>> {on_file} from config file")
    if expl == "PGEex":
        explainer = PGExplainer(model, graph, epochs=epochs, device=device, coeffs=cfg["expl_params"]) # needs 'GNN' model
    else:
        if expl == "CFPG":
            explainer = CFPGExplainer(model, graph, epochs=epochs, device=device, coeffs=cfg["expl_params"])
        elif expl == "CFPGv2":
            conv = cfg["expl_params"]["conv"]
            explainer = CFPGv2(model, graph, conv=conv, epochs=epochs, device=device, coeffs=cfg["expl_params"], verbose=verbose)       
        elif expl == "CFGNN":
            explainer = CFGNNExplainer(model, graph, coeffs=cfg["expl_params"])

        # baseline explainers    
        elif expl == "1hop":
            explainer = OneHopExplainer(model, graph, device=device)
        elif expl == "perfEx":
            explainer = PerfectExplainer(model, graph, device=device)

    return explainer




#### to load CF-GNN model
#if EXPLAINER == "CF-GNN":
#    # need dense-normalized adjacency matrix for GCNSynthetic model
#    v = torch.ones(edge_index.size(1))
#    s = (graph.num_nodes,graph.num_nodes)
#    dense_index = torch.sparse_coo_tensor(indices=edge_index, values=v, size=s).to_dense()
#    norm_adj = normalize_adj(dense_index)

#elif EXPLAINER == "CF-GNN":
#   explainer = PCFExplainer(model, graph, norm_adj, epochs=epochs, device=device, coeffs=cfg["expl_params"])