import tensorflow as tf
import tensorflow.contrib.slim as slim

from ray.rllib.models import Model
from ray.rllib.models.misc import normc_initializer


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
