import os
from neurocuts import *

import ray
from ray.tune import *


def run_neurocuts(config, reporter):
    random.seed(1)
    rules = load_rules_from_file(config["rules"])
    neuro_cuts = NeuroCuts(rules, gamma=config["gamma"], reporter=reporter)
    neuro_cuts.train()


if __name__ == "__main__":
    ray.init()
    run_experiments({
        "neurocuts": {
            "run": run_neurocuts,
            "config": {
                "rules": grid_search([
                    os.path.abspath("classbench/acl1_1000"),
                    os.path.abspath("classbench/acl1_10K"),
                    os.path.abspath("classbench/acl1_100K"),
                ]),
                "gamma": grid_search([0.99, 1.0]),
            },
        },
    })