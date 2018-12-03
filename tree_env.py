import collections
import numpy as np
from gym.spaces import Tuple, Box, Discrete

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models.preprocessors import TupleFlatteningPreprocessor

from tree import Tree, load_rules_from_file
from hicuts import HiCuts


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
            max_actions_per_episode=1000,
            leaf_value_fn="hicuts",
            q_learning=False):

        if leaf_value_fn == "hicuts":
            self.leaf_value_fn = lambda rules: HiCuts(rules).get_depth()
            assert not q_learning, "configuration not supported"
        elif leaf_value_fn is None:
            self.leaf_value_fn = lambda rules: 0
        else:
            raise ValueError("Unknown value fn: {}".format(leaf_value_fn))

        self.order = order
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.onehot_state = onehot_state
        self.max_actions_per_episode = max_actions_per_episode
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.q_learning = q_learning
        if q_learning:
            num_actions = 5 * max_cuts_per_dimension
            self.max_cuts_per_dimension = max_cuts_per_dimension
            self.max_children = 2**max_cuts_per_dimension
            self.action_space = Discrete(num_actions)
            if onehot_state:
                observation_space = Tuple([
                    Box(1, self.max_children, (), dtype=np.float32),  # nchild
                    Box(0, 1, (), dtype=np.float32),  # is finished
                    Box(0, 1, (self.max_children, 208), dtype=np.float32)])
            else:
                observation_space = Tuple([
                    Box(1, self.max_children, (), dtype=np.float32),
                    Box(0, 1, (), dtype=np.float32),
                    Box(0, 1, (self.max_children, 26), dtype=np.float32)])
            self.preprocessor = TupleFlatteningPreprocessor(observation_space)
            self.observation_space = self.preprocessor.observation_space
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
        if self.q_learning:
            state = [self._zeros() for _ in range(self.max_children)]
            state[0] = node.get_state()
            return self.preprocessor.transform([1, 0, state])
        else:
            return node.get_state()

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
        return self.preprocessor.transform(
            [max(1, len(children)), finished, state])

    def step(self, action_dict):
        if self.order == "dfs":
            assert len(action_dict) == 1  # one at a time processing

        new_children = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            if np.isscalar(action):
                cut_dimension = int(action) % 5
                cut_num = int(action) // self.max_cuts_per_dimension
                action = [cut_dimension, cut_num]
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
            while node and self.tree.is_leaf(node):
                node = self.tree.get_next_node()
            nodes_remaining = self.tree.nodes_to_cut

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
                self.num_actions > self.max_actions_per_episode):
            if self.q_learning:
                # terminate the root agent last always to preserve the stats
                obs[0] = self._encode_child_state(self.tree.root)
                rew[0] = -1
            else:
                zero_state = self._zeros()
                rew = self.compute_rewards()
                obs = {node_id: zero_state for node_id in rew.keys()}
                info = {node_id: {} for node_id in rew.keys()}
            info[0] = {
                "tree_depth": self.tree.get_depth(),
                "nodes_remaining": len(nodes_remaining),
                "num_splits": self.num_actions,
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
        rew = {
            node_id: -depth
            for (node_id, depth) in depth_to_go.items()
                if node_id in self.child_map
        }
        return rew
