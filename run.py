import sys, os, time, subprocess, random

from neurocuts import *
from hicuts import *
from hypercuts import *
from efficuts import *
from cutsplit import *

def exe_cmd(cmd):
    #print "\t", cmd
    subprocess.call(cmd, shell=True)

def sync():
    exe_cmd("rsync -r * netx3:~/xinjin/neurocuts")

def run_neurocuts():
    random.seed(1)
    rules = load_rules_from_file("classbench/acl1_100")
    neuro_cuts = NeuroCuts(rules)
    neuro_cuts.train()

def run_hicuts():
    for i in ["100", "500", "1000"]:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = HiCuts(rules)
        cuts.train()

def run_hypercuts():
    for i in ["100", "500", "1000"]:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = HyperCuts(rules)
        cuts.train()

def run_efficuts():
    for i in ["100", "500", "1000"]:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = EffiCuts(rules)
        cuts.train()

def run_cutsplit():
    for i in ["100"]:
        print(i)
        rules = load_rules_from_file("classbench/acl1_%s" % i)
        cuts = CutSplit(rules)
        cuts.train()

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print("Usage")
        print("  tool.py sync")
        print("  tool.py expr")
        sys.exit()

    if sys.argv[1] == "sync":
        sync()
    elif sys.argv[1] == "expr":
        #run_neurocuts()
        #run_hicuts()
        #run_hypercuts()
        #run_efficuts()
        run_cutsplit()
    else:
        print("Not supported option")
