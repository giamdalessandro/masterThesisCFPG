import subprocess

script_cmd  = "/home/zascerta/virtEnvs/XAI-cuda117/bin/python3 src/explain.py "

DATASETS = ["syn1","syn2","syn3","syn4"]
CONVS = ["GCN","GAT"]



script_args = "-E CFPGv2 -D syn4 -e 10 --conv GAT"
cmd = script_cmd + script_args
returned_value = subprocess.call(cmd, shell=True)  # returns the exit code in unix

print('returned value:', returned_value)