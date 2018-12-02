import argparse
import collections
import numpy as np
import os
from gym.spaces import Tuple, Box, Discrete

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import Model, ModelCatalog
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from tree import Tree, load_rules_from_file
from hicuts import HiCuts

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--num-workers", type=int, default=0)


class TreeEnv(MultiAgentEnv):
    def __init__(
            self,
            rules_file,
            leaf_threshold=16,
            onehot_state=True,
            mode="dfs",
            max_cuts_per_dimension=5,
            max_actions_per_episode=1000,
            leaf_value_fn="hicuts",
            encode_next_state=False):

        if leaf_value_fn == "hicuts":
            self.leaf_value_fn = lambda rules: HiCuts(rules).get_depth()
            assert not encode_next_state, "configuration not supported"
        elif leaf_value_fn is None:
            self.leaf_value_fn = lambda rules: 0
        else:
            raise ValueError("Unknown value fn: {}".format(leaf_value_fn))

        self.mode = mode
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.onehot_state = onehot_state
        self.max_actions_per_episode = max_actions_per_episode
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.encode_next_state = encode_next_state
        if encode_next_state:
            num_actions = 5 * max_cuts_per_dimension
            self.max_children = 2**max_cuts_per_dimension
            self.action_space = Discrete(num_actions)
            if onehot_state:
                self.observation_space = Tuple([
                    Discrete(max_children),  # num valid children
                    Box(0, 1, (208, self.max_children), dtype=np.float32)])
            else:
                self.observation_space = Tuple([
                    Discrete(max_children),
                    Box(0, 1, (26, self.max_children), dtype=np.float32)])
        else:
            self.action_space = Tuple(
                [Discrete(5), Discrete(max_cuts_per_dimension)])
            if onehot_state:
                self.observation_space = Box(0, 1, (208,), dtype=np.float32)
            else:
                self.observation_space = Box(0, 1, (26,), dtype=np.float32)

    def reset(self):
        self.num_actions = 0
        self.tree = Tree(
            self.rules, self.leaf_threshold, onehot_state=self.onehot_state)
        self.node_map = {
            self.tree.root.id: self.tree.root,
        }
        self.child_map = {}
        return {
            self.tree.root.id: self._encode_state(self.tree.root)
        }

    def _zeros(self):
        if self.onehot_state:
            zeros = [0] * 208
        else:
            zeros = [0] * 26
        return zeros

    def _encode_state(self, node):
        if self.encode_next_state:
            state = [self._zeros() for _ in range(self.max_children)]
            state[0] = node.get_state()
            return [0, state]
        else:
            return node.get_state()

    def _encode_child_state(self, node):
        assert self.encode_next_state
        children = self.child_map[node]
        assert len(children) < self.max_children, children
        state = [self._zeros() for _ in range(self.max_children)]
        for i, c in enumerate(children):
            state[i] = c.get_state()
        return [len(children), state]

    def step(self, action_dict):
        if self.mode == "dfs":
            assert len(action_dict) == 1  # one at a time processing

        new_children = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            cut_dimension, cut_num = self.action_tuple_to_cut(node, action)
            children = self.tree.cut_node(node, cut_dimension, int(cut_num))
            self.num_actions += 1
            num_leaf = 0
            for c in children:
                self.node_map[c.id] = c
                if not self.tree.is_leaf(c):
                    new_children.append(c)
                else:
                    num_leaf += 1
            self.child_map[node_id] = [c.id for c in children]

        if self.mode == "bfs":
            nodes_remaining = new_children
        else:
            node = self.tree.get_current_node()
            while node and self.tree.is_leaf(node):
                node = self.tree.get_next_node()
            nodes_remaining = self.tree.nodes_to_cut

        if (not nodes_remaining or
                self.num_actions > self.max_actions_per_episode):
            rew = self.compute_rewards()
            if self.encode_next_state:
                obs = {
                    node_id: self._encode_child_state(self.node_map[node_id])
                    for node_id in rew.keys()
                }
            else:
                zero_state = np.zeros_like(self.observation_space.sample())
                obs = {node_id: zero_state for node_id in rew.keys()}
            infos = {node_id: {} for node_id in rew.keys()}
            infos[0] = {
                "tree_depth": self.tree.get_depth(),
                "nodes_remaining": len(nodes_remaining),
                "num_splits": self.num_actions,
            }
            return obs, rew, {"__all__": True}, infos
        else:
            if self.mode == "dfs":
                needs_split = [self.tree.get_current_node()]
            else:
                needs_split = new_children
            return (
                {s.id: s.get_state() for s in needs_split},
                {s.id: 0 for s in needs_split},
                {"__all__": False},
                {s.id: {} for s in needs_split})

    def action_tuple_to_cut(self, node, action):
        cut_dimension = action[0]
        range_left = node.ranges[cut_dimension*2]
        range_right = node.ranges[cut_dimension*2+1]
        cut_num = min(2**(action[1] + 1), range_right - range_left)
        return (cut_dimension, cut_num)

    def compute_rewards(self):
        if self.encode_next_state:
            return self._compute_rewards_1step()
        else:
            return self._compute_rewards_agg()

    def _compute_rewards_1step(self):
        rewards = {}
        for node_id, node in self.node_map.items():
            if node_id in self.child_map:
                rewards[node_id] = -1
        return rewards

    def _compute_rewards_agg(self):
        depth_to_go = collections.defaultdict(int)
        num_updates = 1
        while num_updates > 0:
            num_updates = 0
            for node_id, node in self.node_map.items():
                if node_id not in depth_to_go:
                    if self.tree.is_leaf(node):
                        depth_to_go[node_id] = 0  # is leaf
                    elif node_id not in self.child_map:
                        depth_to_go[node_id] = 0  # no children
                    else:
                        depth_to_go[node_id] = self.leaf_value_fn(node.rules)
                if node_id in self.child_map:
                    max_child_depth = 1 + max(
                        [depth_to_go[c] for c in self.child_map[node_id]])
                    if max_child_depth > depth_to_go[node_id]:
                        depth_to_go[node_id] = max_child_depth
                        num_updates += 1
        rew = {
            node_id: -depth
            for (node_id, depth) in depth_to_go.items()
                if node_id in self.child_map
        }
        return rew


def on_episode_end(info):
    episode = info["episode"]
    info = episode.last_info_for(0)
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
    else:
        info["tree_depth_valid"] = float("nan")
    episode.custom_metrics.update(info)


class MinChildQFunc(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        obs = input_dict["obs"]


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env(
        "tree_env", lambda env_config: TreeEnv(
            env_config["rules"],
            onehot_state=env_config["onehot_state"],
            encode_next_state=env_config["encode_next_state"],
            mode=env_config["mode"]))

    if args.run in ["DQN", "APEX"]:
        ModelCatalog.register_custom_model("min_child_q_func", MinChildQFunc)
        extra_config = {
            "model": {
                "custom_model": "min_child_q_func",
            },
            "hiddens": [],  # don't postprocess the action scores
            "dueling": False,
            "double_q": False,
            "train_batch_size": 64,
        }
    else:
        extra_config = {}

    run_experiments({
        "neurocuts-env":  {
            "run": args.run,
            "env": "tree_env",
            "config": {
                "num_workers": args.num_workers,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
                },
                "env_config": dict({
                    "encode_next_state": args.run in ["DQN", "APEX"],
                    "rules": os.path.abspath("classbench/acl1_500"),
                    "mode": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": grid_search(["hicuts", None]),
                }, **extra_config),
            },
        },
    })
