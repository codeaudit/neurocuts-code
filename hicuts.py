import math

from tree import *

class HiCuts(object):
    def __init__(self, rules):
        # hyperparameters
        self.leaf_threshold = 16                        # number of rules in a leaf
        self.spfac = 4                                  # space estimation
        self.max_cut = 32

        # set up
        self.rules = rules

    # HiCuts heuristic to cut a dimeision
    def select_action(self, tree, node):
        # select a dimension
        cut_dimension = 0
        max_distinct_components_count = -1
        for i in range(5):
            distinct_components = set()
            for rule in node.rules:
                left = max(rule.ranges[i*2], node.ranges[i*2])
                right = min(rule.ranges[i*2+1], node.ranges[i*2+1])
                distinct_components.add((left, right))
            if max_distinct_components_count < len(distinct_components):
                max_distinct_components_count = len(distinct_components)
                cut_dimension = i

        # compute the number of cuts
        range_left = node.ranges[cut_dimension*2]
        range_right = node.ranges[cut_dimension*2+1]
        cut_num = min(
            max(4, int(math.sqrt(len(node.rules)))),
            range_right - range_left)
        while True:
            sm_C = cut_num
            range_per_cut = math.ceil((range_right - range_left) / cut_num)
            for rule in node.rules:
                rule_range_left = rule.ranges[cut_dimension*2]
                rule_range_right = rule.ranges[cut_dimension*2+1]
                sm_C += (rule_range_right - range_left - 1) // range_per_cut - \
                    (rule_range_left - range_left) // range_per_cut
            if sm_C < self.spfac * len(node.rules) and \
                cut_num * 2 <= range_right - range_left and \
                cut_num * 2 <= self.max_cut:
                cut_num *= 2
            else:
                break
        return (cut_dimension, cut_num)

    def train(self):
        tree = Tree(self.rules, self.leaf_threshold)
        node = tree.get_current_node()
        while not tree.is_finish():
            if tree.is_leaf(node):
                node = tree.get_next_node()
                continue

            cut_dimension, cut_num = self.select_action(tree, node)
            tree.cut_current_node(cut_dimension, cut_num)
            node = tree.get_current_node()
        print(tree.get_depth())
        #print(tree)

def test0():
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    cuts = HiCuts(rules)
    cuts.train()

def test1():
    rules = load_rules_from_file("rules/acl1_200")
    cuts = HiCuts(rules)
    cuts.train()

if __name__ == "__main__":
    test1()
