import math

class RuleSet:
    def __init__(self, rules):
        self.rules = rules
        self.rules_in_set = set(rules)

    def __len__(self):
        return len(self.rules)

    def __eq__(self, other):
        return self.rules_in_set == set(other.rules_in_set)

    def get_rules(self):
        return self.rules

    def intersect(self, dimension, range_left, range_right):
        intersected_rules = []
        for rule in self.rules:
            if rule.is_intersect(dimension, range_left, range_right):
                intersected_rules.append(rule)

        return RuleSet(intersected_rules)

    def prune(self, ranges):
        new_rules = []
        rule_len = len(self.rules)
        for i in range(rule_len - 1):
            rule = self.rules[rule_len - 1 - i]
            flag = False
            for j in range(0, rule_len - 1 - i):
                high_priority_rule = self.rules[j]
                if rule.is_covered_by(high_priority_rule, ranges):
                    flag = True
                    break
            if not flag:
                new_rules.append(rule)
        new_rules.append(self.rules[0])
        new_rules.reverse()

        return RuleSet(new_rules)

    def distinct_components_count(self, ranges, i):
        distinct_components = set()
        for rule in self.rules:
            left = max(rule.ranges[i * 2], ranges[i * 2])
            right = min(rule.ranges[i * 2 + 1], ranges[i * 2 + 1])
            distinct_components.add((left, right))
        return len(distinct_components)

    def compute_cuts(self, cut_dimension, range_left, range_right, cut_num):
        ret = cut_num
        range_per_cut = math.ceil((range_right - range_left) / cut_num)
        for rule in self.rules:
            rule_range_left = max(rule.ranges[cut_dimension * 2],
                    range_left)
            rule_range_right = min(rule.ranges[cut_dimension * 2 + 1],
                    range_right)
            ret += (rule_range_right - range_left - 1) // range_per_cut - \
                   (rule_range_left - range_left) // range_per_cut + 1

        return ret
