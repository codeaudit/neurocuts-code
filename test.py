from tree import *
from neurocuts import *
from hicuts import *

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
    node.compact_ranges()
    print(node)

    print("========== tree ==========")
    rules = []
    rules.append(Rule([0, 10, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([0, 10, 10, 20, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 0, 10, 0, 1, 0, 1, 0, 1]))
    rules.append(Rule([10, 20, 10, 20, 0, 1, 0, 1, 0, 1]))
    tree = Tree(rules, 1)
    tree.root.compact_ranges()
    tree.cut_current_node(0, 2)
    tree.print_layers()

    tree.cut_current_node(1, 2)
    tree.get_next_node()
    tree.get_next_node()
    tree.cut_current_node(1, 2)
    tree.print_layers()

    print("========== print tree ==========")
    print(tree)

    print("========== load rule ==========")
    rules = load_rules_from_file("classbench/acl1_20")
    for rule in rules:
        print(rule)

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

if __name__ == "__main__":
    #test_tree()
    #test_neurocuts()
    #test_hicuts()
