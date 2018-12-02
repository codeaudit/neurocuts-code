import collections
import numpy as np
import os
from gym.spaces import Tuple, Box, Discrete

from ray.rllib.env import MultiAgentEnv
import ray
from ray.tune import run_experiments
from ray.tune.registry import register_env

from tree import *

DIMENSIONS_TO_CUT = 5
MAX_ACTIONS_PER_EPISODE = 1000


class TreeEnv(MultiAgentEnv):
    def __init__(
            self,
            rules_file,
            leaf_threshold=16,
            onehot_state=False,
            max_cuts_per_dimension=5):
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.onehot_state = onehot_state
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.action_space = Tuple(
            [Discrete(DIMENSIONS_TO_CUT),
             Discrete(max_cuts_per_dimension)])
        if onehot_state:
            self.observation_space = Box(0, 1, (208,), dtype=np.float32)
        else:
            self.observation_space = Box(0, 1, (26,), dtype=np.float32)

    def reset(self):
        print("new episode")
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
        needs_split = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            cut_dimension, cut_num = self.action_tuple_to_cut(node, action)
            children = self.tree.cut_node(node, cut_dimension, cut_num)
            self.num_actions += 1
            num_leaf = 0
            for c in children:
                self.node_map[c.id] = c
                if not self.tree.is_leaf(c):
                    needs_split.append(c)
                else:
                    num_leaf += 1
#            print(
#                "split", node_id, "depth", node.depth, "action", action,
#                "->", len(children), "num leaf", num_leaf,
#                "num actions", self.num_actions)
            self.child_map[node_id] = [c.id for c in children]

        if not needs_split or self.num_actions > MAX_ACTIONS_PER_EPISODE:
            print("Remaining nodes on terminate", len(needs_split))
            rew = self.compute_rewards()
            zero_state = np.zeros_like(self.observation_space.sample())
            obs = {node_id: zero_state for node_id in rew.keys()}
            infos = {node_id: {} for node_id in rew.keys()}
            return obs, rew, {"__all__": True}, infos
        else:
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
                    depth_to_go[node_id] = 0
                if node_id in self.child_map:
                    max_child_depth = 1 + max(
                        [depth_to_go[c] for c in self.child_map[node_id]])
                    if max_child_depth > depth_to_go[node_id]:
                        depth_to_go[node_id] = max_child_depth
                        num_updates += 1
        rew = {node_id: -depth for (node_id, depth) in depth_to_go.items()}
#        print("rewards", rew)
        return rew


if __name__ == "__main__":
    ray.init()
    register_env(
        "tree_env", lambda env_config: TreeEnv(env_config["rules"]))
    run_experiments({
        "neurocuts": {
            "run": "PPO",
            "env": "tree_env",
            "config": {
                "num_workers": 0,
                "batch_mode": "complete_episodes",
                "simple_optimizer": True,
                "env_config": {
                    "rules": os.path.abspath("classbench/acl1_100"),
                },
            },
        },
    })
