import random
import math

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

    def push(self, transition):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class CutsNet(nn.Module):
    def __init__(self, action_size):
        super(CutsNet, self).__init__()
        self.fc1 = nn.Linear(26, 50)
        self.fc2 = nn.Linear(50, 50)
        self.fc3 = nn.Linear(50, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class NeuroCuts(object):
    def __init__(self, rules):
        # hyperparameters
        self.N = 1000                     # maximum number of episodes
        self.t_train = 10               # training interval
        self.C = 3                      # target model copy interval
        self.gamma = 0.99               # reward discount factor
        self.epsilon_start = 1.0        # exploration start rate
        self.epsilon_end = 0.1          # exploration end rate
        self.alpha = 0.1                # learning rate
        self.batch_size = 64            # batch size
        self.replay_memory_size = 10000 # replay memory size
        self.cuts_per_dimension = 5     # cuts per dimension
        self.action_size = 5 * self.cuts_per_dimension            # action size
        self.leaf_threshold = 16        # number of rules in a leaf

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

    def select_action(self, state, n):
        """
        For test evaluation, you should turn off the random exploration.
        """
        epsilon_threshold = max(self.epsilon_end,
                                self.epsilon_end + \
                                    (self.epsilon_start - self.epsilon_end) * \
                                    math.exp(-1. * n / self.N * 2))
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

        # expected q values
        max_next_q_values = []
        for children in batch_children:
            non_final_mask = torch.tensor(
                [not tree.is_leaf(child) for child in children],
                dtype=torch.uint8)
            non_final_children = [child.get_state() for child in children if not tree.is_leaf(child)]

            children_q_values = torch.zeros(len(children))
            if len(non_final_children) > 0:
                non_final_children = torch.cat(non_final_children)
                children_q_values[non_final_mask] = \
                    self.target_net(non_final_children).max(1)[0].detach()
            max_next_q_values.append(children_q_values.min().view(1, 1))
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
        while n < self.N:
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
                    self.replay_memory.push((node, action, children, reward))
                    break
                else:
                    reward = torch.tensor([[-1.]])
                    self.replay_memory.push((node, action, children, reward))

                # update parameters
                if t % self.t_train == 0:
                    #print("time ", t, tree.get_depth())
                    if t % (self.C * self.t_train) == 0:
                        self.target_net.load_state_dict(
                            self.policy_net.state_dict())
                    self.optimize_model(tree)
                node = tree.get_current_node()
                t += 1

            # store the tree with min depth
            if min_tree is None or tree.get_depth() < min_tree.get_depth():
                min_tree = tree

            if n % 20 == 0:
                print("episode ", n, self.batch_count, min_tree.get_depth())
                #if min_tree.get_depth() < 15:
                #    print(min_tree)

            # next episode
            n += 1

        print(min_tree.get_depth())
        print(min_tree)

def test0():
    random.seed(1)
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    neuro_cuts = NeuroCuts(rules)
    neuro_cuts.train()

def test1():
    random.seed(1)
    rules = load_rules_from_file("rules/acl1_20")
    neuro_cuts = NeuroCuts(rules)
    neuro_cuts.train()

if __name__ == "__main__":
    test0()
