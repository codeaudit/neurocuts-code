import argparse
import collections
import numpy as np
import os
from gym.spaces import Tuple, Box, Discrete
import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.env import MultiAgentEnv
from ray.rllib.models import Model, ModelCatalog
import ray
from ray import tune
from ray.rllib.models.misc import normc_initializer
from ray.rllib.models.preprocessors import TupleFlatteningPreprocessor
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env

from tree import Tree, load_rules_from_file
from hicuts import HiCuts

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="PPO")
parser.add_argument("--env", type=str, default="acl1_100")
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
            self.max_cuts_per_dimension = max_cuts_per_dimension
            self.max_children = 2**max_cuts_per_dimension
            self.action_space = Discrete(num_actions)
            if onehot_state:
                observation_space = Tuple([
                    Discrete(self.max_children + 1),  # num valid children
                    Box(0, 1, (self.max_children, 208), dtype=np.float32)])
            else:
                observation_space = Tuple([
                    Discrete(self.max_children + 1),
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
        if self.encode_next_state:
            state = [self._zeros() for _ in range(self.max_children)]
            state[0] = node.get_state()
            return self.preprocessor.transform([1, state])
        else:
            return node.get_state()

    def _encode_child_state(self, node):
        assert self.encode_next_state
        children = self.child_map[node.id]
        assert len(children) <= self.max_children, children
        state = [self._zeros() for _ in range(self.max_children)]
        for i, c in enumerate(children):
            state[i] = self.node_map[c].get_state()
        return self.preprocessor.transform([len(children), state])

    def step(self, action_dict):
        if self.mode == "dfs":
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
                {s.id: self._encode_state(s) for s in needs_split},
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
    """A Q function that internally takes the argmin over nodes in the state.

    The input states are (num_nodes, node_states).

    When num_nodes == 1, the returned action scores are simply that of a
    normal q function over node_states[0]. This Q output is hence valid for
    selecting an action via argmax.

    When num_nodes > 1, the returned action scores are uniformly the min
    over the max Q values of all children. This Q output is only valid for
    computing the Q target value = min child max a' Q(s{t+1}{child}, a').
    """

    def _build_layers_v2(self, input_dict, num_outputs, options):
        num_nodes = input_dict["obs"][0]  # shape [BATCH]
        node_states = input_dict["obs"][1]  # shape [BATCH, MAX_CHILD, STATE]
        max_children = node_states.get_shape().as_list()[-2]

        # shape [BATCH * MAX_CHILD, STATE]
        flat_shape = [-1, node_states.get_shape().as_list()[-1]]
        nodes_flat = tf.reshape(node_states, flat_shape)

        # Mask for invalid nodes (use tf.float32.min for stability)
        action_mask = tf.cast(
            tf.sequence_mask(num_nodes, max_children), tf.float32)
        flat_inf_mask = tf.reshape(
            tf.maximum(tf.log(action_mask), tf.float32.min), flat_shape)

        # push flattened node states through the Q network
        last_layer = nodes_flat
        for i, size in enumerate([256, 256]):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.tanh,
                scope=label)

        # shape [BATCH * MAX_CHILD, NUM_ACTIONS]
        action_scores = slim.fully_connected(
            last_layer,
            num_outputs,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out")

        # shape [BATCH, MAX_CHILD, NUM_ACTIONS]
        masked_scores = tf.reshape(
            action_scores + flat_inf_mask,
            node_states.get_shape().as_list()[:-1] + [num_outputs])

        # case 1: emit [BATCH, NUM_ACTIONS <- actual scores for node 0]
        action_out1 = masked_scores[:, 0, :]

        # case 2: emit [BATCH, NUM_ACTIONS <- uniform; min over all nodes]
        child_min_max = tf.reduce_min(
            tf.reduce_max(masked_scores, axis=2), axis=1)
        action_out2 = tf.tile(tf.expand_dims(child_min_max), num_outputs)

        output = tf.where(tf.equal(num_nodes, 1), action_out1, action_out2)
        return output, output


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init()

    register_env(
        "tree_env", lambda env_config: TreeEnv(
            env_config["rules"],
            onehot_state=env_config["onehot_state"],
            encode_next_state=env_config["encode_next_state"],
            leaf_value_fn=env_config["leaf_value_fn"],
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
            "leaf_value_fn": None,
        }
    elif args.run == "PPO":
        extra_config = {
            "entropy_coeff": 0.01,
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
                    "rules": os.path.abspath("classbench/{}".format(args.env)),
                    "mode": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": grid_search([None, "hicuts"]),
                }, **extra_config),
            },
        },
    })
