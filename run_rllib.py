import argparse
import os
import glob
import json

from ray.rllib.models import ModelCatalog
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from tree_env import TreeEnv
from q_func import MinChildQFunc


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--gpu", type=bool, default=False)
parser.add_argument("--env", type=str, default="acl1_100")
parser.add_argument("--num-workers", type=int, default=0)


def on_episode_end(info):
    """Report tree custom metrics"""
    episode = info["episode"]
    info = episode.last_info_for(0)
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
        info["num_nodes_valid"] = info["num_nodes"]
        info["num_splits_valid"] = info["num_splits"]
        info["mean_split_size_valid"] = info["mean_split_size"]
        info["bytes_per_rule_valid"] = info["bytes_per_rule"]
        info["memory_access_valid"] = info["memory_access"]
        pid = info["rules_file"].split("/")[-1]
        with open(os.path.expanduser("~/valid_trees-{}.txt".format(pid)), "a") as f:
            f.write(json.dumps(info))
            f.write("\n")
    else:
        info["tree_depth_valid"] = float("nan")
        info["num_nodes_valid"] = float("nan")
        info["num_splits_valid"] = float("nan")
        info["mean_split_size_valid"] = float("nan")
        info["bytes_per_rule_valid"] = float("nan")
        info["memory_access_valid"] = float("nan")
    del info["rules_file"]
    episode.custom_metrics.update(info)


def erase_done_values(info):
    """Hack: set dones=False so that Q-backup in a tree works."""
    samples = info["samples"]
    samples["dones"] = [False for _ in samples["dones"]]


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env(
        "tree_env", lambda env_config: TreeEnv(
            env_config["rules"],
            onehot_state=env_config["onehot_state"],
            q_learning=env_config["q_learning"],
            leaf_value_fn=env_config["leaf_value_fn"],
            order=env_config["order"],
            max_actions_per_episode=env_config["max_actions"],
            cut_weight=env_config["cut_weight"],
            partition_enabled=env_config["partition_enabled"]))

    extra_config = {}
    extra_env_config = {}

    q_learning = args.run in ["DQN", "APEX"]
    if q_learning:
        ModelCatalog.register_custom_model("min_child_q_func", MinChildQFunc)
        extra_config = {
            "model": {
                "custom_model": "min_child_q_func",
            },
            "hiddens": [],  # don't postprocess the action scores
            "train_batch_size": 32,
            "dueling": False,
            "double_q": False,
            "batch_mode": "truncate_episodes",
        }
        if args.run == "APEX":
            extra_config.update({
                "train_batch_size": 512,
                "buffer_size": 50000,
                "learning_starts": 5000,
                "target_network_update_freq": 50000,
                "timesteps_per_iteration": 5000,
                "min_iter_time_s": 5,
            })
        extra_env_config = {
            "leaf_value_fn": None,
        }
    elif args.run == "PPO":
        extra_config = {
            "entropy_coeff": 0.01,
            "sample_batch_size": 5000,
            "train_batch_size": 5000,
        }
    elif args.run == "IMPALA":
        extra_config = {
            "sample_batch_size": 5000,
            "train_batch_size": 5000,
        }

    run_experiments({
        "neurocuts-env-part":  {
            "run": args.run,
            "env": "tree_env",
            "stop": {
                "timesteps_total": 1000000,
            },
            "config": dict({
                "num_gpus": 1 if args.gpu else 0,
                "num_workers": args.num_workers,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
                    "on_sample_end": tune.function(erase_done_values)
                        if q_learning else None,
                },
                "env_config": dict({
                    "q_learning": q_learning,
#                    "rules": os.path.abspath("classbench/acl1_1000"),
                    "partition_enabled": grid_search([False, True]),
                    "rules": grid_search(
                        [os.path.abspath(x) for x in glob.glob("classbench/*10000")]),
                    "order": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": None, #grid_search([None, "hicuts"]),
                    "max_actions": 5000,
                    "cut_weight": 0, #grid_search([0, 0.001]),
                }, **extra_env_config),
            }, **extra_config),
        },
    })
