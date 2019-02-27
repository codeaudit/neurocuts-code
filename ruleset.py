import time
import numpy as np
import math


class RuleSet:
    def __init__(self, rules):
        self.rules = rules
        self.rules_data = np.array([r.ranges for r in rules], dtype=np.int64)
        self._rules_in_set = None

    @property
    def rules_in_set(self):
        if self._rules_in_set is None:
            self._rules_in_set = set(self.rules)

        return self._rules_in_set

    def __len__(self):
        return len(self.rules)

    def __eq__(self, other):
        return self.rules_in_set == other.rules_in_set

    def get_rules(self):
        return self.rules

    def intersect(self, dimension, range_left, range_right):
        start = time.time()
        intersected_rules = []
        for rule in self.rules:
            if rule.is_intersect(dimension, range_left, range_right):
                intersected_rules.append(rule)
        norm_time = time.time() - start
        start2 = time.time()

        np_result = self.rules_data[
            np.logical_not(
                np.logical_or(
                    range_left >= self.rules_data[:, dimension*2+1],
                    range_right <= self.rules_data[:, dimension*2]))]
        np_time = time.time() - start2
        print("speedup", norm_time / np_time)
        assert len(np_result) == len(intersected_rules)

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
