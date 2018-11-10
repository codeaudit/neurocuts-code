import sys

from tree import *
from neurocuts import *
from hicuts import *
from hypercuts import *

def test_tree():
    print("========== rule ==========")
    rule = Rule([0, 10, 0, 10, 10, 20, 0, 1, 0, 1])
    print(rule)
    print("True", rule.is_intersect(2, 0, 11))
    print("False", rule.is_intersect(2, 0, 10))
    print("False", rule.is_intersect(2, 20, 21))
    print("True", rule.is_intersect_multi_dimension([0, 10, 0, 10, 0, 11, 0, 1, 0, 1]))
    print("False", rule.is_intersect_multi_dimension([0, 10, 0, 10, 0, 10, 0, 1, 0, 1]))
    print("False", rule.is_intersect_multi_dimension([0, 10, 0, 10, 20, 21, 0, 1, 0, 1]))

    print("========== node ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 0]))
    rules.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 0]))
    rules.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 0]))
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node = Node(0, ranges, rules, 1)
    print(node)

    print("========== tree single-dimensional cuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)
    tree.refinement_region_compaction(tree.root)
    print(tree.root)
    tree.cut_current_node(0, 2)
    tree.print_layers()

    tree.cut_current_node(1, 2)
    tree.get_next_node()
    tree.get_next_node()
    tree.cut_current_node(1, 2)
    tree.print_layers()

    print("========== tree multi-dimensional cuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)
    tree.refinement_region_compaction(tree.root)
    tree.cut_current_node_multi_dimension([0, 1, 2, 3, 4], [2, 2, 1, 1, 1])
    tree.print_layers()

    print("========== print tree ==========")
    print(tree)

    print("========== load rule ==========")
    rules = load_rules_from_file("classbench/acl1_20")
    for rule in rules:
        print(rule)

def test_refinements():
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)

    print("========== node merging ==========")
    rules1 = []
    rules1.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 1]))
    rules2 = [rule for rule in rules1]
    rules3 = [rule for rule in rules1]
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node1 = Node(0, [0, 100, 0, 100, 0, 1000, 0, 1000, 0, 1000], rules1, 1)
    node2 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules2, 1)
    node3 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules3[1:], 1)
    print("True", tree.refinement_node_merging(node1, node2))
    print("False", tree.refinement_node_merging(node1, node3))

    node1 = Node(0, [0, 100, 0, 100, 0, 1000, 0, 1000, 0, 1000], rules1, 1)
    node2 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules2, 1)
    node3 = Node(1, [0, 100, 100, 200, 0, 1000, 0, 1000, 0, 1000], rules3, 1)
    node4 = Node(1, [0, 100, 200, 300, 0, 1000, 0, 1000, 0, 1000], rules3, 1)
    tree.update_tree(tree.root, [node1, node2, node3, node4])
    print(node1)
    print(node3)

    print("========== rule overlay ==========")
    rule1 = Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1])
    rule2 = Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1])
    rule3 = Rule([0, 12, 0, 10, 10, 20, 10, 15, 0, 1])
    rule4 = Rule([0, 9, 0, 10, 10, 20, 10, 15, 0, 1])
    rule5 = Rule([0, 9, 0, 10, 10, 20, 10, 15, 0, 2])
    ranges = [0, 10, 0, 10, 10, 20, 10, 15, 0, 2]
    print("True", rule1.is_covered_by(rule2, ranges))
    print("True", rule1.is_covered_by(rule3, ranges))
    print("False", rule1.is_covered_by(rule4, ranges))

    node1 = Node(0, ranges, [rule1, rule2, rule3, rule4, rule5], 1)
    tree.refinement_rule_overlay(node1)
    print(node1)

    ranges = [0, 9, 0, 10, 10, 20, 10, 15, 0, 1]
    print("True", rule1.is_covered_by(rule4, ranges))

    node1 = Node(0, ranges, [rule1, rule2, rule3, rule4, rule5], 1)
    tree.refinement_rule_overlay(node1)
    print(node1)

    print("========== region compaction ==========")
    rules1 = []
    rules1.append(Rule([0, 10, 0, 10, 10, 20, 10, 15, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 20, 30, 0, 1]))
    rules1.append(Rule([0, 100, 0, 100, 0, 100, 40, 50, 0, 1]))
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    node1 = Node(0, ranges, rules1, 1)
    print(node1)
    tree.refinement_region_compaction(node1)
    print(node1)

    print("========== rule pushup ==========")
    rule1 = Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1])
    rule2 = Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1])
    rule3 = Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1])
    rule4 = Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1])
    ranges = [0, 1000, 0, 1000, 0, 1000, 0, 1000, 0, 1000]
    tree = Tree([rule1, rule2, rule3, rule4], 1)
    node1 = tree.create_node(1, ranges.copy(), [rule1, rule2, rule3], 2)
    node2 = tree.create_node(2, ranges.copy(), [rule2, rule4], 2)
    tree.update_tree(tree.root, [node1, node2])
    node3 = tree.create_node(3, ranges.copy(), [rule1, rule2, rule3], 3)
    node4 = tree.create_node(4, ranges.copy(), [rule1, rule2], 3)
    tree.update_tree(node1, [node3, node4])
    tree.depth = 3
    tree.print_layers()

    tree.refinement_rule_pushup()
    tree.print_layers()


def test_neurocuts():
    print("========== neurocuts ==========")
    random.seed(1)
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    neuro_cuts = NeuroCuts(rules)
    neuro_cuts.train()

def test_hicuts():
    print("========== hicuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    cuts = HiCuts(rules)
    cuts.train()

def test_hypercuts():
    print("========== hypercuts ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    cuts = HyperCuts(rules)
    cuts.leaf_threshold = 1
    cuts.train()

if __name__ == "__main__":
    #test_tree()
    test_refinements()
    #test_neurocuts()
    #test_hicuts()
    #test_hypercuts()
