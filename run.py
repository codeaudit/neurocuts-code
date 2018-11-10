from neurocuts import *
from hicuts import *
from hypercuts import *

def run_neurocuts():
    random.seed(1)
    rules = load_rules_from_file("classbench/acl1_100")
    neuro_cuts = NeuroCuts(rules)
    neuro_cuts.train()

def run_hicuts():
    rules = load_rules_from_file("classbench/acl1_200")
    cuts = HiCuts(rules)
    cuts.train()

def run_hypercuts():
    rules = load_rules_from_file("../classbench/acl1_100K")
    cuts = HyperCuts(rules)
    cuts.train()

if __name__ == "__main__":
    #run_neurocuts()
    #run_hicuts()
    run_hypercuts()
