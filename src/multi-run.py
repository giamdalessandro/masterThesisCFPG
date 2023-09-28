import os
#import subprocess
from subprocess import PIPE, run
from tqdm import tqdm

DATASETS = ["syn1","syn2","syn3","syn4"] #
CONVS = ["GAT","GCN"] #,"pGCN"
EXPLAINER = "CFPGv2" # "CFPG", "CFPGv2", "PGEex"
EPOCHS = 50

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
    "syn1" : "--opt Adam --reg-ent 0.1 --reg-cf 1.0 --reg-size 0.1 --heads 3 --hid-gcn 20 --add-att 0.0", # --temp [1.0,1.0]
    "syn2" : "--opt Adam --reg-ent 0.1 --reg-cf 1.0 --reg-size 0.1 --heads 3 --hid-gcn 20 --add-att 0.0", # --temp [1.0,1.0]
    "syn3" : "--opt Adam --reg-ent 0.1 --reg-cf 1.0 --reg-size 0.1 --heads 3 --hid-gcn 20 --add-att 0.0", # --temp [1.0,1.0]
    "syn4" : "--opt Adam --reg-ent 0.1 --reg-cf 1.0 --reg-size 0.1 --heads 3 --hid-gcn 20 --add-att 0.0", # --temp [1.0,1.0] 
}


script_cmd = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py" #src/explain.py"
rid = 0
n_exps = len(SEEDS)*len(DATASETS)*len(CONVS)
seeds_bar = tqdm(range(n_exps), desc=f"[multi-run]> experiments", colour="green", disable=False)
for c in CONVS:
#    if c == "GAT": es = 5
#    elif c == "GCN": es = 10
#    for curr in ENT_COEFFS:
#for e in [50, 100]:
    for s in SEEDS:
        for d in DATASETS:
        #for d in (d_bar := tqdm(DATASETS, desc=f"[seed {s:03}]> datasets... ", colour="green", disable=False)):
            script_args = f" -E {EXPLAINER} -D {d} -e {EPOCHS} --conv {c} {params[d]} --seed {s} -es 10 "
            suffix_args = f"--prefix rAllOne-Sparsemax0-thres01-allEdges-3{c}-Estop10-fullSparse-dec64 --log"
            args = script_args + suffix_args
            cmd = script_cmd + args
            #command = [cmd, args]

            tqdm.write(f"\n------------------------------ run id: {rid} curr-> {EXPLAINER} - {d} - seed {s}\n")
            result = run(cmd, capture_output=True, shell=True)
            for o in (result.stdout).decode("utf-8").split("\n"):
                tqdm.write(o)

            tqdm.write(f">> run exit-status: {result.returncode}")
            rid += 1

            seeds_bar.update()

seeds_bar.close()
print("[multi-runs]> ...DONE") #, result.returncode)