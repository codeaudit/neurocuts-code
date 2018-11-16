import random
import math
import datetime

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tree import *

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition, tree):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        n, a, c, r = transition
        non_final_mask = torch.tensor(
            [not tree.is_leaf(child) for child in c],
            dtype=torch.uint8)
        c = (torch.cat([n.get_state() for n in c]), non_final_mask)
        self.memory[self.position] = (n, a, c, r)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CutsNet(nn.Module):
    def __init__(self, action_size):
        super(CutsNet, self).__init__()
        self.fc1 = nn.Linear(208, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuroCuts(object):
    def __init__(self, rules, gamma=0.99, reporter=None):
        # hyperparameters
        self.N = 1000                   # maximum number of episodes
        self.t_train = 10               # training interval
        self.C = 3                      # target model copy interval
        self.gamma = gamma              # reward discount factor
        self.epsilon_start = 1.0        # exploration start rate
        self.epsilon_end = 0.1          # exploration end rate
        self.alpha = 0.1                # learning rate
        self.batch_size = 64            # batch size
        self.replay_memory_size = 100000 # replay memory size
        self.cuts_per_dimension = 5     # cuts per dimension
        self.action_size = 5 * self.cuts_per_dimension  # action size
        self.leaf_threshold = 16        # number of rules in a leaf
        self.reporter = reporter

        # set up
        self.replay_memory = ReplayMemory(self.replay_memory_size)

        self.policy_net = CutsNet(self.action_size)
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.batch_count = 0
        self.criterion = nn.MSELoss()

        self.target_net = CutsNet(self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.rules = rules
        self.test = False

    def select_action(self, state, n):
        """
        For test evaluation, you should turn off the random exploration.
        """
        epsilon_threshold = max(self.epsilon_end,
                                self.epsilon_end + \
                                    (self.epsilon_start - self.epsilon_end) * \
                                    math.exp(-1. * n / self.N * 2))
        if self.test:
            epsilon_threshold = 0.0
        if random.random() > epsilon_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_size)]],
                dtype=torch.long)

    def optimize_model(self, tree):
        if len(self.replay_memory.memory) < self.batch_size:
            return
        self.batch_count += 1
        transitions = self.replay_memory.sample(self.batch_size)
        batch_node, batch_action, batch_children, batch_reward = zip(*transitions)
        batch_state = torch.cat([node.get_state() for node in batch_node])
        batch_action = torch.cat(batch_action)
        batch_reward = torch.cat(batch_reward)

        # current q values
        current_q_values = self.policy_net(batch_state).gather(1, batch_action)

        flat_children = []
        for children, _ in batch_children:
            flat_children.extend(children)
        target_values = self.target_net(torch.stack(flat_children))

        # expected q values
        max_next_q_values = []
        i = 0
        for children, non_final_mask in batch_children:
            children_q_values = torch.zeros(len(children))
            children_q_values[non_final_mask] = \
                target_values[i:i+len(children)].max(1)[0].detach()[non_final_mask]
            max_next_q_values.append(children_q_values.min().view(1, 1))
            i += len(children)
        max_next_q_values = torch.cat(max_next_q_values)
        expected_q_values = batch_reward + self.gamma * max_next_q_values

        # compute loss, and update parameters
        loss = self.criterion(current_q_values, expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def action_index_to_cut(self, node, action):
        cut_dimension = action.item() // self.cuts_per_dimension
        range_left = node.ranges[cut_dimension*2]
        range_right = node.ranges[cut_dimension*2+1]
        cut_num = min(2**(action.item() % self.cuts_per_dimension + 1),
            range_right - range_left)
        return (cut_dimension, cut_num)

    def train(self):
        min_tree = None
        n = 0
        while True:
            if n % 20 == 0:
                self.test = True
            else:
                self.test = False
            # build a new tree
            tree = Tree(self.rules, self.leaf_threshold)
            node = tree.get_current_node()
            t = 0
            while not tree.is_finish():
                if tree.is_leaf(node):
                    node = tree.get_next_node()
                    continue

                action = self.select_action(node.get_state(), n)
                cut_dimension, cut_num = self.action_index_to_cut(node, action)
                children = tree.cut_current_node(cut_dimension, cut_num)
                if tree.get_depth() > 22 and t > 1000:
                    reward = torch.tensor([[-100.]])
                    self.replay_memory.push((node, action, children, reward), tree)
                    break
                else:
                    reward = torch.tensor([[-1.]])
                    self.replay_memory.push((node, action, children, reward), tree)

                # update parameters
                if t % self.t_train == 0:
                    if t % (self.C * self.t_train) == 0:
                        self.target_net.load_state_dict(
                            self.policy_net.state_dict())
                    self.optimize_model(tree)
                node = tree.get_current_node()
                t += 1

            # store the tree with min depth
            if min_tree is None or not self.test:
                if min_tree is None or tree.get_depth() < min_tree.get_depth():
                    min_tree = tree

            if self.test:
                if self.reporter:
                    self.reporter(
                        timesteps_total=n,
                        test_depth=tree.get_depth(),
                        train_depth=min_tree.get_depth())
                print(datetime.datetime.now(),
                    "Episode:", n,
                    "Batch:", self.batch_count,
                    "Test Depth:", tree.get_depth(),
                    "Train Depth:", min_tree.get_depth())
                #if min_tree.get_depth() < 15:
                #    print(min_tree)

            # next episode
            n += 1

        print(datetime.datetime.now(), "Depth:", min_tree.get_depth())
        print(datetime.datetime.now(), "Tree:\n", min_tree)
