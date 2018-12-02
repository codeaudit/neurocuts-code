import collections
import numpy as np
import os
from gym.spaces import Tuple, Box, Discrete

from ray.rllib.env import MultiAgentEnv
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from tree import Tree, load_rules_from_file
from hicuts import HiCuts


class TreeEnv(MultiAgentEnv):
    def __init__(
            self,
            rules_file,
            leaf_threshold=16,
            onehot_state=True,
            mode="dfs",
            max_cuts_per_dimension=5,
            max_actions_per_episode=1000,
            leaf_value_fn="hicuts"):

        if leaf_value_fn == "hicuts":
            self.leaf_value_fn = lambda rules: HiCuts(rules).get_depth()
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
            self.tree.root.id: self.tree.root.get_state(),
        }

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
        rew = {node_id: -depth for (node_id, depth) in depth_to_go.items()}
        return rew


def on_episode_end(info):
    episode = info["episode"]
    info = episode.last_info_for(0)
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
    else:
        info["tree_depth_valid"] = float("nan")
    episode.custom_metrics.update(info)


if __name__ == "__main__":
    ray.init()

    register_env(
        "tree_env", lambda env_config: TreeEnv(
            env_config["rules"],
            onehot_state=env_config["onehot_state"],
            mode=env_config["mode"]))

    run_experiments({
        "neurocuts-env":  {
            "run": "PPO",
            "env": "tree_env",
            "config": {
                "num_workers": 24,
                "batch_mode": "complete_episodes",
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
                },
                "env_config": {
                    "rules": os.path.abspath("classbench/acl1_500"),
                    "mode": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": grid_search(["hicuts", None]),
                },
            },
        },
    })
