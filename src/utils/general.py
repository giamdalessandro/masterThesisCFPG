import argparse
from torch.cuda import device, is_available, get_device_name


def cuda_device_check(device, switch: bool=False, verbose: bool=False): 
    """Check CUDA availabilty and print device, if any."""
    if device == "cuda":
        if is_available() and switch:
            cuda_dev = device("cuda")
            if verbose: 
                print(">> cuda available", cuda_dev)
                print(">> device: ", get_device_name(cuda_dev),"\n")
    else: 
        if verbose: print(">> Device 'cpu' selected")


def parser_add_args(parser: argparse.ArgumentParser):
    """Add arguments to argparser."""
    parser.add_argument("--explainer", "-E", type=str, default="CFPG",
                        choices=["PGEex","CFPG","CFPGv2","1hop","perfEx"])
    parser.add_argument("--dataset", "-D", type=str, default="syn1", 
                        choices=['syn1','syn2','syn3','syn4'], help="Dataset used for training")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Number of explainer epochs")
    parser.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--train-nodes", default=False, action=argparse.BooleanOptionalAction,
                        help="Whether to explain original train nodes")
    parser.add_argument("--opt", "-o", type=str, default="base", 
                        choices=["Adam","SGD","SGDm"], help="learning optimizer")
    parser.add_argument("--early-stop", "-es", type=int, default=10, help="Early stopping (no. epochs)")

    # to test gnn conv, may move it to cfg.json
    parser.add_argument("--conv", "-c", type=str, default="base", 
                        choices=["base","GCN","GAT","pGCN","VAE"], help="Explainer graph convolution")
    parser.add_argument("--heads", type=int, default=1, help="Attention heads (if conv is 'GAT')")
    parser.add_argument("--hid-gcn", type=int, default=20, help="Graph convolution hidden dimension.")
    parser.add_argument("--add-att", type=float, default=0.0, help="Attention coeff")
    parser.add_argument("--reg-ent", type=float, default=0.0, help="Entropy loss coeff")
    parser.add_argument("--reg-size", type=float, default=0.0, help="Size loss coeff")
    parser.add_argument("--reg-cf", type=float, default=0.0, help="Pred loss coeff")

    # other arguments
    parser.add_argument("--prefix", type=str, default="",
                        help="Notes on the current run")
    parser.add_argument("--log", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to store run logs")
    parser.add_argument("--roc", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to plot ROC curve")
    parser.add_argument('--plot-expl', default=False, action=argparse.BooleanOptionalAction, 
                        help="Plot some of the computed explanation")
    parser.add_argument("--store-adv", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to store adv samples")
    parser.add_argument("--device", "-d", default="cpu", help="Running device, 'cpu' or 'cuda'")
    parser.add_argument("--verbose", default=False, action=argparse.BooleanOptionalAction, 
                        help="Whether to plot infos.")

    return parser