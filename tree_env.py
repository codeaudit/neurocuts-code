import collections
import numpy as np
from gym.spaces import Tuple, Box, Discrete, Dict

from ray.rllib.env import MultiAgentEnv

from tree import Tree, load_rules_from_file
from hicuts import HiCuts


NUM_PART_LEVELS = 6  # 2%, 4%, 8%, 16%, 32%, 64%


class TreeEnv(MultiAgentEnv):
    """Two modes: q_learning and on-policy.

    If q_learning=True, each cut in the tree is recorded as -1 reward and a
    transition is returned on each step.
    
    In on-policy mode, we aggregate rewards at the end of the episode and
    assign each cut its reward based on the policy performance (actual depth).

    Both modes are modeled as a multi-agent environment. Each "cut" in the tree
    is an action taken by a different agent. All the agents share the same
    policy.
    """

    def __init__(
            self,
            rules_file,
            leaf_threshold=16,
            onehot_state=True,
            order="dfs",
            max_cuts_per_dimension=5,
            max_actions_per_episode=5000,
            max_depth=100,
            partition_enabled=False,
            leaf_value_fn=None,
            penalty_fn=None,
            q_learning=False,
            cut_weight=0.0):

        if leaf_value_fn == "hicuts":
            self.leaf_value_fn = lambda rules: HiCuts(rules).get_depth()
            assert not q_learning, "configuration not supported"
        elif leaf_value_fn == "len":
            self.leaf_value_fn = lambda rules: len(rules)
            assert not q_learning, "configuration not supported"
        elif leaf_value_fn == "constant":
            self.leaf_value_fn = lambda rules: 10
            assert not q_learning, "configuration not supported"
        elif leaf_value_fn is None:
            self.leaf_value_fn = lambda rules: 0
        else:
            raise ValueError("Unknown value fn: {}".format(leaf_value_fn))

        if penalty_fn is None:
            self.penalty_fn = lambda node: 0
        elif penalty_fn == "correct_useless":
            def penalty(node):
                if node.is_useless():
                    return -1
                else:
                    return 0

            self.penalty_fn = penalty
        elif penalty_fn == "useless_nodes":
            def penalty(node):
                if node.is_useless():
                    return 10
                else:
                    return 0

            self.penalty_fn = penalty
        else:
            raise ValueError("Unknown penalty fn: {}".format(penalty_fn))

        self.order = order
        self.cut_weight = cut_weight
        self.rules_file = rules_file
        self.partition_enabled = partition_enabled
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.onehot_state = onehot_state
        self.max_actions_per_episode = max_actions_per_episode
        self.max_depth = max_depth
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.q_learning = q_learning
        self.max_cuts_per_dimension = max_cuts_per_dimension
        if q_learning:
            num_actions = 5 * max_cuts_per_dimension
            self.max_children = 2**max_cuts_per_dimension
            if self.partition_enabled:
                num_actions += 5 * NUM_PART_LEVELS
            self.action_space = Discrete(num_actions)
            if onehot_state:
                self.observation_space = Tuple([
                    Box(1, self.max_children, (), dtype=np.float32),  # nchild
                    Box(0, 1, (), dtype=np.float32),  # is finished
                    Box(0, 1, (self.max_children, 278), dtype=np.float32)])
            else:
                self.observation_space = Tuple([
                    Box(1, self.max_children, (), dtype=np.float32),
                    Box(0, 1, (), dtype=np.float32),
                    Box(0, 1, (self.max_children, 36), dtype=np.float32)])
        else:
            if self.partition_enabled:
                num_part_levels = NUM_PART_LEVELS
            else:
                num_part_levels = 0
            self.num_part_levels = num_part_levels
            self.action_space = Tuple(
                [Discrete(5), Discrete(max_cuts_per_dimension + num_part_levels)])
            if onehot_state:
                x = 278
            else:
                x = 36
            self.observation_space = Dict({
                "real_obs": Box(0, 1, (x,), dtype=np.float32),
                "action_mask": Box(
                    0, 1,
                    (5 + max_cuts_per_dimension + num_part_levels,),
                    dtype=np.float32),
            })

    def reset(self):
        self.num_actions = 0
        self.exceeded_max_depth = []
        self.tree = Tree(
            self.rules, self.leaf_threshold, refinements = {
                "node_merging"      : True,
                "rule_overlay"      : True,
                "region_compaction" : False,
                "rule_pushup"       : False,
                "equi_dense"        : False,
            }, onehot_state=self.onehot_state)
        self.node_map = {
            self.tree.root.id: self.tree.root,
        }
        self.child_map = {}
        return {
            self.tree.root.id: self._encode_state(self.tree.root)
        }

    def _zeros(self):
        if self.onehot_state:
            zeros = [0] * 278
        else:
            zeros = [0] * 36
        return {
            "real_obs": zeros,
            "action_mask": [1] * (5 + self.max_cuts_per_dimension + self.num_part_levels),
        }

    def _encode_state(self, node):
        if self.q_learning:
            state = [self._zeros() for _ in range(self.max_children)]
            state[0] = node.get_state()
            return [1, 0, state]
        else:
            if node.depth > 1:
                action_mask = (
                    [1] * (5 + self.max_cuts_per_dimension) +
                    [0] * self.num_part_levels)
            else:
                assert node.depth == 1, node.depth
                action_mask = (
                    [1] * (5 + self.max_cuts_per_dimension) +
                    [1] * self.num_part_levels)
            return {
                "real_obs": node.get_state(),
                "action_mask": action_mask,
            }

    def _encode_child_state(self, node):
        assert self.q_learning
        children = self.child_map[node.id]
        assert len(children) <= self.max_children, children
        state = [self._zeros() for _ in range(self.max_children)]
        finished = 1
        for i, c in enumerate(children):
            child = self.node_map[c]
            state[i] = child.get_state()
            if not self.tree.is_leaf(child):
                finished = 0
        return [max(1, len(children)), finished, state]

    def step(self, action_dict):
        if self.order == "dfs":
            assert len(action_dict) == 1  # one at a time processing

        new_children = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            orig_action = action
            if np.isscalar(action):
                if int(action) >= 5 * self.max_cuts_per_dimension:
                    assert self.partition_enabled, action
                    action = int(action) - 5 * self.max_cuts_per_dimension
                    part_num = action % 5
                    part_size = action // 5
                    action = [part_num, part_size]
                    partition = True
                else:
                    partition = False
                    cut_dimension = int(action) % 5
                    cut_num = int(action) // 5
                    action = [cut_dimension, cut_num]
            else:
                if action[1] >= self.max_cuts_per_dimension:
                    assert self.partition_enabled, action
                    partition = True
                    action[1] -= self.max_cuts_per_dimension
                else:
                    partition = False

            if partition:
                children = self.tree.partition_node(node, action[0], action[1])
            else:
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

        if self.order == "bfs":
            nodes_remaining = new_children
        else:
            node = self.tree.get_current_node()
            while node and (self.tree.is_leaf(node) or node.depth > self.max_depth):
                node = self.tree.get_next_node()
                if node and node.depth > self.max_depth:
                    self.exceeded_max_depth.append(node)
            nodes_remaining = self.tree.nodes_to_cut + self.exceeded_max_depth

        if self.q_learning:
            obs, rew, done, info = {}, {}, {}, {}
            for node_id, action in action_dict.items():
                if node_id == 0:
                    continue
                obs[node_id] = self._encode_child_state(self.node_map[node_id])
                rew[node_id] = -1
                done[node_id] = True
                info[node_id] = {}
        else:
            obs, rew, done, info = {}, {}, {}, {}

        if (not nodes_remaining or
                self.num_actions > self.max_actions_per_episode or
                self.tree.get_current_node() is None):
            if self.q_learning:
                # terminate the root agent last always to preserve the stats
                obs[0] = self._encode_child_state(self.tree.root)
                rew[0] = -1
            else:
                zero_state = self._zeros()
#                rew = self.compute_rewards(
#                    self.cut_weight if not nodes_remaining else 0.0)
                rew = self.compute_rewards(self.cut_weight)
                obs = {node_id: zero_state for node_id in rew.keys()}
                info = {node_id: {} for node_id in rew.keys()}
            result = self.tree.compute_result()
            rules_remaining = set()
            largest_node_remaining = 0
            for n in nodes_remaining:
                for r in n.rules:
                    rules_remaining.add(str(r))
                largest_node_remaining = max(largest_node_remaining, len(n.rules))
            info[0] = {
                "bytes_per_rule": result["bytes_per_rule"],
                "memory_access": result["memory_access"],
                "exceeded_max_depth": len(self.exceeded_max_depth),
                "tree_depth": self.tree.get_depth(),
                "nodes_remaining": len(nodes_remaining),
                "largest_node_remaining": largest_node_remaining,
                "rules_remaining": len(rules_remaining),
                "num_nodes": len(self.node_map),
                "useless_fraction": float(len(
                    [n for n in self.node_map.values() if n.is_useless()])) / len(self.node_map),
                "partition_fraction": float(len(
                    [n for n in self.node_map.values() if n.is_partition()])) / len(self.node_map),
                "mean_split_size": np.mean(
                    [len(x) for x in self.child_map.values()]),
                "num_splits": self.num_actions,
                "rules_file": self.rules_file,
            }
            return obs, rew, {"__all__": True}, info
        else:
            if self.order == "dfs":
                needs_split = [self.tree.get_current_node()]
            else:
                needs_split = new_children
            obs.update({s.id: self._encode_state(s) for s in needs_split})
            rew.update({s.id: 0 for s in needs_split})
            done.update({"__all__": False})
            info.update({s.id: {} for s in needs_split})
            return obs, rew, done, info

    def action_tuple_to_cut(self, node, action):
        cut_dimension = action[0]
        range_left = node.ranges[cut_dimension*2]
        range_right = node.ranges[cut_dimension*2+1]
        cut_num = max(
            2,
            min(2**(action[1] + 1), range_right - range_left))
        return (cut_dimension, cut_num)

    def compute_rewards(self, cut_weight):
        depth_to_go = collections.defaultdict(int)
        cuts_to_go = collections.defaultdict(int)
        num_updates = 1
        while num_updates > 0:
            num_updates = 0
            for node_id, node in self.node_map.items():
                if node_id not in depth_to_go:
                    if self.tree.is_leaf(node):
                        depth_to_go[node_id] = 0  # is leaf
                        cuts_to_go[node_id] = 0
                    elif node_id not in self.child_map:
                        depth_to_go[node_id] = 0  # no children
                        cuts_to_go[node_id] = 0
                    else:
                        depth_to_go[node_id] = self.leaf_value_fn(node.rules)
                        cuts_to_go[node_id] = 0
                if node_id in self.child_map:
                    if self.node_map[node_id].is_partition():
                        max_child_depth = sum(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    else:
                        max_child_depth = 1 + max(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    if max_child_depth > depth_to_go[node_id]:
                        depth_to_go[node_id] = max_child_depth
                        num_updates += 1
                    sum_child_cuts = len(self.child_map[node_id]) + sum(
                        [cuts_to_go[c] for c in self.child_map[node_id]])
                    if sum_child_cuts > cuts_to_go[node_id]:
                        cuts_to_go[node_id] = sum_child_cuts
                        num_updates += 1
        rew = {
            node_id:
                - depth
                - (cut_weight * cuts_to_go[node_id])
                - self.penalty_fn(self.node_map[node_id])
            for (node_id, depth) in depth_to_go.items()
                if node_id in self.child_map
        }
        return rew
