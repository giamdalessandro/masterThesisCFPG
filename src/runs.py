import subprocess

script_cmd  = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "

DATASETS = ["syn1","syn2","syn3","syn4"]
CONVS = ["pGCN","GCN","GAT"]

ENT_COEFFS = [0.1, 0.5, 1.0, 2.0]
#SIZE_COEFFS = [0.1, 0.01, 0.001]
#CF_COEFFS = [0.5, 1.0, 2.0, 5.0]


rid = 0
for c in CONVS:
    for d in DATASETS:
        #for curr in ENT_COEFFS:
        script_args = f"-E CFPGv2 -D {d} -e 20 --conv {c} "
        suffix_args = f"--prefix all-rBest-20-tests --log"
        cmd = script_cmd + script_args + suffix_args

        print("\n\n------------------------------ run id:", rid, f"curr-> {c} - {d}\n")
        returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
        rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)