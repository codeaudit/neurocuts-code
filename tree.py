import math
import re

import torch

class Rule:
    def __init__(self, ranges):
        # each range is left inclusive and right exclusive, i.e., [left, right)
        self.ranges = ranges
        self.names = ["src_ip", "dst_ip", "src_port", "dst_port", "proto"]

    def is_intersect(self, dimension, left, right):
        return not (left >= self.ranges[dimension*2+1] or \
            right <= self.ranges[dimension*2])

    def __str__(self):
        result = ""
        for i in range(len(self.names)):
            result += "%s:[%d, %d) " % (self.names[i],
                self.ranges[i*2], self.ranges[i*2+1])
        return result

def load_rules_from_file(file_name):
    rules = []
    rule_fmt = re.compile(r'^@(\d+).(\d+).(\d+).(\d+)/(\d+) '\
        r'(\d+).(\d+).(\d+).(\d+)/(\d+) ' \
        r'(\d+) : (\d+) ' \
        r'(\d+) : (\d+) ' \
        r'(0x[\da-fA-F]+)/(0x[\da-fA-F]+) ' \
        r'(.*?)')
    for idx, line in enumerate(open(file_name)):
        elements = line[1:-1].split('\t')
        line = line.replace('\t',' ')

        sip0, sip1, sip2, sip3, sip_mask_len, \
        dip0, dip1, dip2, dip3, dip_mask_len, \
        sport_begin, sport_end, \
        dport_begin, dport_end, \
        proto, proto_mask = \
        (eval(rule_fmt.match(line).group(i)) for i in range(1, 17))

        sip0 = (sip0 << 24) | (sip1 << 16) | (sip2 << 8) | sip3
        sip_begin = sip0 & (~((1 << (32 - sip_mask_len)) - 1))
        sip_end = sip0 | ((1 << (32 - sip_mask_len)) - 1)

        dip0 = (dip0 << 24) | (dip1 << 16) | (dip2 << 8) | dip3
        dip_begin = dip0 & (~((1 << (32 - dip_mask_len)) - 1))
        dip_end = dip0 | ((1 << (32 - dip_mask_len)) - 1)

        if proto_mask == 0xff:
            proto_begin = proto
            proto_end = proto
        else:
            proto_begin = 0
            proto_end = 0xff

        rules.append(Rule([sip_begin, sip_end+1,
            dip_begin, dip_end+1,
            sport_begin, sport_end+1,
            dport_begin, dport_end+1,
            proto_begin, proto_end+1]))
    return rules

class Node:
    def __init__(self, id, ranges, rules, depth):
        self.id = id
        self.ranges = ranges
        self.rules = rules
        self.depth = depth
        self.children = []
        self.compute_state()
        self.action = None

    def compute_state(self):
        self.state = []
        for i in range(4):
            for j in range(4):
                self.state.append(((self.ranges[i] - i % 2)  >> (j * 8)) & 0xff)
        for i in range(4):
            for j in range(2):
                self.state.append(((self.ranges[i+4] - i % 2) >> (j * 8)) & 0xff)
        self.state.append(self.ranges[8])
        self.state.append(self.ranges[9] - 1)
        self.state = [[i / 256 for i in self.state]]
        self.state = torch.tensor(self.state)

    def compact_ranges(self):
        new_ranges = self.rules[0].ranges.copy()
        for rule in self.rules[1:]:
            for i in range(len(self.ranges)//2):
                new_ranges[i*2] = min(new_ranges[i*2], rule.ranges[i*2])
                new_ranges[i*2+1] = max(new_ranges[i*2+1], rule.ranges[i*2+1])
        for i in range(len(self.ranges)//2):
            self.ranges[i*2] = max(new_ranges[i*2], self.ranges[i*2])
            self.ranges[i*2+1] = min(new_ranges[i*2+1], self.ranges[i*2+1])
        self.compute_state()

    def get_state(self):
        return self.state

    def __str__(self):
        result = "ID:%d\tAction:%s\tDepth:%d\tRange:\t%s\nChildren: " % (
            self.id, str(self.action), self.depth, str(self.ranges))
        for child in self.children:
            result += str(child.id) + " "
        result += "\nRules:\n"
        for rule in self.rules:
            result += str(rule) + "\n"
        return  result

class Tree:
    def __init__(self, rules, leaf_threshold):
        # hyperparameters
        self.leaf_threshold = leaf_threshold

        self.rules = rules
        self.root = Node(0, [0, 2**32, 0, 2**32, 0, 2**16, 0, 2**16, 0, 2**8], rules, 1)
        self.current_node = self.root
        self.nodes_to_cut = [self.root]
        self.depth = 1
        self.node_count = 1

    def get_depth(self):
        return self.depth

    def get_current_node(self):
        return self.current_node

    def is_leaf(self, node):
        return len(node.rules) <= self.leaf_threshold

    def is_finish(self):
        return len(self.nodes_to_cut) == 0

    def cut_current_node(self, cut_dimension, cut_num):
        self.depth = max(self.depth, self.current_node.depth + 1)
        node = self.current_node
        node.action = (cut_dimension, cut_num)
        range_left = node.ranges[cut_dimension*2]
        range_right = node.ranges[cut_dimension*2+1]
        range_per_cut = math.ceil((range_right - range_left) / cut_num)

        children = []
        for i in range(cut_num):
            child_ranges = node.ranges.copy()
            child_ranges[cut_dimension*2] = range_left + i * range_per_cut
            child_ranges[cut_dimension*2+1] = min(range_right,
                range_left + (i+1) * range_per_cut)

            child_rules = []
            for rule in node.rules:
                if rule.is_intersect(cut_dimension,
                    child_ranges[cut_dimension*2],
                    child_ranges[cut_dimension*2+1]):
                    child_rules.append(rule)

            child = Node(self.node_count, child_ranges, child_rules, node.depth + 1)
            children.append(child)
            self.node_count += 1

        node.children.extend(children)
        children.reverse()
        self.nodes_to_cut.pop()
        self.nodes_to_cut.extend(children)
        self.current_node = self.nodes_to_cut[-1]
        return children

    def get_next_node(self):
        self.nodes_to_cut.pop()
        if len(self.nodes_to_cut) > 0:
            self.current_node = self.nodes_to_cut[-1]
        else:
            self.current_node = None
        return self.current_node

    def print_layers(self, layer_num = 5):
        nodes = [self.root]
        for i in range(layer_num):
            if len(nodes) == 0:
                return

            print("Layer", i)
            next_layer_nodes = []
            for node in nodes:
                print(node)
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes

    def __str__(self):
        result = ""
        nodes = [self.root]
        while len(nodes) != 0:
            next_layer_nodes = []
            for node in nodes:
                result += "%d; %s; %s; [" % (
                    node.id, str(node.action), str(node.ranges))
                for child in node.children:
                    result += str(child.id) + " "
                result += "]\n"
                next_layer_nodes.extend(node.children)
            nodes = next_layer_nodes
        return result
