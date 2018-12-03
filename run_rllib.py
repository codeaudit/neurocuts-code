import argparse
import os

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
    else:
        info["tree_depth_valid"] = float("nan")
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
            order=env_config["order"]))

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
        }

    run_experiments({
        "neurocuts-env":  {
            "run": args.run,
            "env": "tree_env",
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
                    "rules": os.path.abspath("classbench/{}".format(args.env)),
                    "order": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": grid_search([None, "hicuts"]),
                }, **extra_env_config),
            }, **extra_config),
        },
    })
