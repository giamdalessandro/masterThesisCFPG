import subprocess

script_cmd  = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "

DATASETS = ["syn1","syn2","syn3","syn4"]
CONVS = ["GCN","GAT"]

CF_COEFFS = [0.1, 0.01, 0.001]
fixed_ent = 1.0
fixed_size = 1.0

rid = 0
for c in CONVS:
    for d in DATASETS:
        for cf in CF_COEFFS:
            script_args = f"-E CFPGv2 -D {d} -e 10 --conv {c} --reg-ent {fixed_ent} --reg-size {fixed_size} --reg-cf {cf} --prefix rcfTest_{cf}_ --log"
            cmd = script_cmd + script_args

            print("\n\n------------------------------ run id:", rid, "\n")
            returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix
            rid += 1

print("\n[runs]> Multi-run DONE...", returned_value)