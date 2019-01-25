import argparse
import os
import glob
import json

from gym.spaces import Tuple, Box, Discrete, Dict
import numpy as np

from ray.rllib.models import ModelCatalog
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
from ray.rllib.agents.ppo.ppo_policy_graph import PPOPolicyGraph

from tree_env import TreeEnv
from q_func import MinChildQFunc
from mask import PartitionMaskModel


parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--env", type=str, default="acl1_100")
parser.add_argument("--partition", type=str, default=None)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--max-agents", type=int, default=1)
parser.add_argument("--redis-address", type=str, default=None)


def on_episode_end(info):
    """Report tree custom metrics"""
    episode = info["episode"]
    info = episode.last_info_for(0)
    if not info:
        info = episode.last_info_for((0, 0))
    pid = info["rules_file"].split("/")[-1]
    out = os.path.abspath(os.path.expanduser("~/valid_trees-{}.txt".format(pid)))
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
        info["num_nodes_valid"] = info["num_nodes"]
        info["num_splits_valid"] = info["num_splits"]
        info["bytes_per_rule_valid"] = info["bytes_per_rule"]
        info["memory_access_valid"] = info["memory_access"]
        with open(out, "a") as f:
            f.write(json.dumps(info))
            f.write("\n")
    else:
        info["tree_depth_valid"] = float("nan")
        info["num_nodes_valid"] = float("nan")
        info["num_splits_valid"] = float("nan")
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
    ray.init(redis_address=args.redis_address)

    register_env(
        "tree_env", lambda env_config: TreeEnv(
            env_config["rules"],
            q_learning=env_config["q_learning"],
            leaf_value_fn=env_config["leaf_value_fn"],
            max_depth=env_config["max_depth"],
            max_actions_per_episode=env_config["max_actions"],
            cut_weight=env_config["cut_weight"],
            partition_mode=env_config["partition_mode"]))

    extra_config = {}
    extra_env_config = {}

    q_learning = args.run in ["DQN", "APEX"]
    if q_learning:
        ModelCatalog.register_custom_model("min_child_q_func", MinChildQFunc)
        extra_config = {
            "model": {
                "custom_model": "min_child_q_func",
            },
            "evaluation_interval": 3,
            "evaluation_num_episodes": 3,
            "hiddens": [],  # don't postprocess the action scores
            "train_batch_size": 32,
            "dueling": False,
            "double_q": False,
            "batch_mode": "truncate_episodes",
        }
        if args.run == "APEX":
            extra_config.update({
                "train_batch_size": 512,
                "buffer_size": 500000,
                "learning_starts": 5000,
                "target_network_update_freq": 50000,
                "timesteps_per_iteration": 5000,
                "min_iter_time_s": 5,
            })
        extra_env_config = {
            "leaf_value_fn": None,
        }
    else:
        ModelCatalog.register_custom_model("mask", PartitionMaskModel)

    if args.run == "PPO":
        extra_config = {
            "model": {
                "custom_model": "mask",
                "fcnet_hiddens": [512, 512],
            },
            "vf_share_layers": False,
            "entropy_coeff": 0.01,
            "sgd_minibatch_size": 100 if args.test else 1000,
            "sample_batch_size": 200 if args.test else 5000,
            "train_batch_size": 200 if args.test else 5000,
        }

    MAX_AGENTS = int(args.max_agents)

    def policy_mapper(agent_id):
        if type(agent_id) is tuple:
            if agent_id[0] >= MAX_AGENTS - 1:
                return "partition_{}".format(MAX_AGENTS - 1)
            else:
                return "partition_{}".format(agent_id[0])
        return "partition_0"

    if args.partition == "top":
        num_part_actions = 5
    else:
        num_part_actions = 0
    a_sp = Tuple(
        [Discrete(5), Discrete(5 + num_part_actions)])
    o_sp = Dict({
        "real_obs": Box(0, 1, (278,), dtype=np.float32),
        "action_mask": Box(
            0, 1,
            (5 + 5 + num_part_actions,),
            dtype=np.float32),
    })
    pol_graphs = {
        "partition_{}".format(i):
            (PPOPolicyGraph, o_sp, a_sp, {}) for i in range(MAX_AGENTS)
    }

    run_experiments({
        "neurocuts_nagents_{}".format(MAX_AGENTS): {
            "run": args.run,
            "env": "tree_env",
            "stop": {
                "timesteps_total": 10000000,
            },
            "config": dict({
                "num_gpus": 0.2 if args.gpu else 0,
                "num_workers": args.num_workers,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
                    "on_sample_end": tune.function(erase_done_values)
                        if q_learning else None,
                },
                "multiagent": {
                    "policy_mapping_fn": tune.function(policy_mapper),
                    "policy_graphs": pol_graphs,
                },
                "env_config": dict({
                    "q_learning": q_learning,
                    "partition_mode": args.partition,
                    "max_depth": 500,
                    "max_actions": 1000 if args.test else 15000,
                    "cut_weight": 0, #grid_search([0, 0.0001, 0.005, 0.001, 0.005, 0.01]),
                    "leaf_value_fn": None,
                    "rules":
                        os.path.abspath("classbench/acl1_seed_1000")
                        if args.test else
                        grid_search([
                            os.path.abspath(x) for x in
                            glob.glob("classbench/*_100000")
                        ]),
#                        grid_search([
#                            os.path.abspath("classbench/acl4_seed_1000"),
#                            os.path.abspath("classbench/fw1_seed_10000"),
#                            os.path.abspath("classbench/fw3_seed_10000"),
#                            os.path.abspath("classbench/fw4_seed_1000"),
#                            os.path.abspath("classbench/fw4_seed_10000"),
#                            os.path.abspath("classbench/ipc1_seed_10000"),
#                            os.path.abspath("classbench/acl2_seed_100000"),
#                            os.path.abspath("classbench/acl3_seed_100000"),
#                            os.path.abspath("classbench/acl4_seed_100000"),
#                            os.path.abspath("classbench/fw1_seed_100000"),
#                            os.path.abspath("classbench/fw3_seed_100000"),
#                            os.path.abspath("classbench/fw4_seed_100000"),
#                            os.path.abspath("classbench/fw5_seed_100000"),
#                            os.path.abspath("classbench/ipc1_seed_100000"),
#                        ]),
                }, **extra_env_config),
            }, **extra_config),
        },
    })
