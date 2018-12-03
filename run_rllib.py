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
            q_learning=False):

        if leaf_value_fn == "hicuts":
            self.leaf_value_fn = lambda rules: HiCuts(rules).get_depth()
            assert not q_learning, "configuration not supported"
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
                zero_state = [1, self._zeros()]
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
            if self.mode == "dfs":
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

        if self.q_learning:
            return self._compute_rewards_1step()
        else:
            return self._compute_rewards_agg()

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
        num_nodes = tf.cast(input_dict["obs"][0], tf.int32)  # shape [BATCH]
        finished = input_dict["obs"][1]  # shape [BATCH]
        node_states = input_dict["obs"][2]  # shape [BATCH, MAX_CHILD, STATE]
        max_children = node_states.get_shape().as_list()[-2]
        num_actions = num_outputs

        # shape [BATCH * MAX_CHILD, STATE]
        flat_shape = [-1, node_states.get_shape().as_list()[-1]]
        nodes_flat = tf.reshape(node_states, flat_shape)

        # Mask for invalid nodes (use tf.float32.min for stability)
        # shape [BATCH, MAX_CHILD]
        action_mask = tf.cast(
            tf.sequence_mask(num_nodes, max_children), tf.float32)
        # shape [BATCH, MAX_CHILD, NUM_ACTIONS]
        action_mask = tf.tile(
            tf.expand_dims(action_mask, 2), [1, 1, num_actions])
        # shape [BATCH * MAX_CHILD, NUM_ACTIONS]
        flat_inf_mask = tf.reshape(
            tf.minimum(-tf.log(action_mask), tf.float32.max),
            [-1, num_actions])

        # push flattened node states through the Q network
        last_layer = nodes_flat
        for i, size in enumerate([256, 256]):
            label = "fc{}".format(i)
            last_layer = slim.fully_connected(
                last_layer,
                size,
                weights_initializer=normc_initializer(1.0),
                activation_fn=tf.nn.relu,
                scope=label)

        # shape [BATCH * MAX_CHILD, NUM_ACTIONS]
        action_scores = slim.fully_connected(
            last_layer,
            num_actions,
            weights_initializer=normc_initializer(0.01),
            activation_fn=None,
            scope="fc_out")

        # shape [BATCH, MAX_CHILD, NUM_ACTIONS]
        masked_scores = tf.reshape(
            action_scores + flat_inf_mask, [-1, max_children, num_actions])

        # case 1: emit [BATCH, NUM_ACTIONS <- actual scores for node 0]
        action_out1 = masked_scores[:, 0, :]

        # case 2: emit [BATCH, NUM_ACTIONS <- uniform; min over all nodes]
        child_min_max = tf.reduce_min(
            tf.reduce_max(masked_scores, axis=2), axis=1)
        action_out2 = tf.tile(
            tf.expand_dims(child_min_max, 1), [1, num_actions])

        output = tf.where(tf.equal(num_nodes, 1), action_out1, action_out2)

        # zero out the Q values for finished nodes (to replace done mask)
        # if you don't do this the Q est will decrease unboundedly!
        output = output * tf.expand_dims(1 - finished, 1)
        return output, output


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
            mode=env_config["mode"]))

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
            "dueling": False,
            "double_q": False,
            "batch_mode": "truncate_episodes",
        }
        extra_env_config = {
            "leaf_value_fn": None,
            "onehot_state": False,
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
                    "mode": "dfs",
                    "onehot_state": True,
                    "leaf_value_fn": grid_search([None, "hicuts"]),
                }, **extra_env_config),
            }, **extra_config),
        },
    })
