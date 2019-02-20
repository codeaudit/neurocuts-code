import time
from hicuts import *

rules = load_rules_from_file("classbench/acl1_10k")
start = time.time()
cuts = HiCuts(rules)
print("Total split time", time.time() - start)
cuts.train()
