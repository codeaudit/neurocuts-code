import time
import numpy as np
import math

# Slow mode -- checks numpy result against Python implementation
DEBUG = False


def python_to_numpy(rules):
    return np.array([r.ranges + [r.priority] for r in rules], dtype=np.int64)


def numpy_to_python(rules_data):
    from tree import Rule
    rules = []
    for rule in rules_data:
        ranges = list(rule[:-1])
        priority = rule[-1]
        rules.append(Rule(priority, ranges))
    return rules


class RuleSet:
    def __init__(self, rules=None, rules_data=None):
        if rules is not None:
            self.rules_data = python_to_numpy(rules)
        elif rules_data is not None:
            self.rules_data = rules_data
        else:
            raise ValueError("One of rules or rules_data must be given")

    def __len__(self):
        return len(self.rules_data)

    def __eq__(self, other):
        return np.array_equal(self.rules_data, other.rules_data)

    def intersect(self, dimension, range_left, range_right):
        if DEBUG:
            rules = numpy_to_python(self.rules_data)
            start = time.time()
            intersected_rules = []
            for rule in rules:
                if rule.is_intersect(dimension, range_left, range_right):
                    intersected_rules.append(rule)
            py_time = time.time() - start
            start2 = time.time()

        np_result = self.rules_data[
            np.logical_not(
                np.logical_or(
                    range_left >= self.rules_data[:, dimension*2+1],
                    range_right <= self.rules_data[:, dimension*2]))]
        if DEBUG:
            np_time = time.time() - start2
            print("intersect speedup", py_time / np_time)
            assert len(np_result) == len(intersected_rules)

        return RuleSet(rules_data=np_result)

    def prune(self, ranges):
        rules = numpy_to_python(self.rules_data)
        new_rules = []
        rule_len = len(rules)
        for i in range(rule_len - 1):
            rule = rules[rule_len - 1 - i]
            flag = False
            for j in range(0, rule_len - 1 - i):
                high_priority_rule = rules[j]
                if rule.is_covered_by(high_priority_rule, ranges):
                    flag = True
                    break
            if not flag:
                new_rules.append(rule)
        new_rules.append(rules[0])
        new_rules.reverse()

        return RuleSet(new_rules)

    def distinct_components_count(self, ranges, i):
        if DEBUG:
            start = time.time()
            rules = numpy_to_python(self.rules_data)
            distinct_components = set()
            for rule in rules:
                left = max(rule.ranges[i * 2], ranges[i * 2])
                right = min(rule.ranges[i * 2 + 1], ranges[i * 2 + 1])
                distinct_components.add((left, right))
            py_time = time.time() - start
            start2 = time.time()

        a = np.maximum(self.rules_data[:, i*2], ranges[i*2])
        b = np.minimum(self.rules_data[:, i*2+1], ranges[i*2+1])
        c = np.stack([a, b], axis=1)
        count = len(np.unique(c, axis=0))
        if DEBUG:
            np_time = time.time() - start2
            print("distinct_components speedup", py_time / np_time)
            assert count == len(distinct_components)

        return count

    def compute_cuts(self, cut_dimension, range_left, range_right, cut_num):
        range_per_cut = math.ceil((range_right - range_left) / cut_num)

        if DEBUG:
            start = time.time()
            rules = numpy_to_python(self.rules_data)
            ret = cut_num
            for rule in rules:
                rule_range_left = max(rule.ranges[cut_dimension * 2],
                        range_left)
                rule_range_right = min(rule.ranges[cut_dimension * 2 + 1],
                        range_right)
                ret += (rule_range_right - range_left - 1) // range_per_cut - \
                       (rule_range_left - range_left) // range_per_cut + 1
            py_time = time.time() - start
            start2 = time.time()

        rule_range_left = np.maximum(
            self.rules_data[:, cut_dimension*2], range_left)
        rule_range_right = np.minimum(
            self.rules_data[:, cut_dimension*2+1], range_right)
        np_ret = (  
            cut_num +
            np.sum((rule_range_right - range_left - 1) // range_per_cut -
                (rule_range_left - range_left) // range_per_cut + 1))
        if DEBUG:
            np_time = time.time() - start2
            print("compute_cuts speedup", py_time / np_time)
            assert np_ret == ret

        return np_ret
