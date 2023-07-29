import os
#import subprocess
from subprocess import PIPE, run
from tqdm import tqdm

DATASETS = ["syn1","syn2","syn3","syn4"] #
CONVS = ["GAT","GCN"] #,"pGCN"
EXPLAINER = "CFPGv2" # "CFPG", "CFPGv2", "PGEex"
EPOCHS = 100

# returns a list of 8 random small integers between 0 and 255
def get8RandomBytesFromOS():
    """Get random integers (in the interval [0,255]) from OS to be used as seeds."""
    r8 = os.urandom(8)  # official OS entropy source
    byteCodes = list(map(ord, r8.decode('Latin-1')))  # type conversion
    return byteCodes

#ENT_COEFFS = [0.1, 0.5, 1.0, 2.0, 5.0]
#SIZE_COEFFS = [0.1, 0.01, 0.001, 0.0005]
#CF_COEFFS = [0.1, 0.5, 1.0, 2.0, 5.0]
#NUM_HEADS = [3, 5, 8]
SEEDS = get8RandomBytesFromOS()[:4]

params = {
    "syn1" : "--reg-ent 1.0 --reg-cf 1.0 --reg-size 0.1 --opt Adam --heads 3 --hid-gcn 20",
    "syn2" : "--reg-ent 1.0 --reg-cf 1.0 --reg-size 0.1 --opt Adam --heads 3 --hid-gcn 20",
    "syn3" : "--reg-ent 1.0 --reg-cf 1.0 --reg-size 0.1 --opt Adam --heads 3 --hid-gcn 20",
    "syn4" : "--reg-ent 1.0 --reg-cf 1.0 --reg-size 0.1 --opt Adam --heads 3 --hid-gcn 20",
}


script_cmd = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py" #src/explain.py"
rid = 0
#for c in CONVS:
#    for curr in ENT_COEFFS:

seeds_bar = tqdm(SEEDS, desc=f"[multi-run]> experiments", colour="yellow", disable=False)
for s in SEEDS:
    for d in (d_bar := tqdm(DATASETS, desc=f"[seed {s:03}]> datasets... ", colour="green", disable=False)):
        script_args = f" -E {EXPLAINER} -D {d} -e {EPOCHS} --conv GAT {params[d]} --seed {s} "
        suffix_args = f"--prefix rAll1-SparsemaxMono-thresTest-3GATreal-CatAtt"
        args = script_args + suffix_args
        cmd = script_cmd + args
        #command = [cmd, args]

        tqdm.write(f"\n------------------------------ run id: {rid} curr-> {EXPLAINER} - {d} - seed {s}\n")
        result = run(cmd, capture_output=True, shell=True)
        for o in (result.stdout).decode("utf-8").split("\n"):
            tqdm.write(o)

        rid += 1

    seeds_bar.update()

print("\n[runs]> Multi-run DONE...", result.returncode)