import time
import numpy as np
import math

# Slow mode -- checks numpy result against Python implementation
DEBUG = True
DEBUG_EQUALITY = False

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


from numba import jit


@jit(nopython=True, cache=True)
def prune_numba(rules_data, ranges):
    bool_1d = np.array([False] * len(rules_data))
    ranges = np.array(ranges)
    rule_len = len(rules_data)
    i = 0
    lim = rule_len - 1
    while i < lim:
        rule = rules_data[rule_len - 1 - i]
        flag = False
        j = 0
        lim_j = rule_len - 1 - i
        while j < lim_j:
            other = rules_data[j]
            is_covered = True
            k = 0
            while k < 5:
                if (np.maximum(rule[k*2], ranges[k*2]) < \
                        np.maximum(other[k*2], ranges[k*2])) or \
                        (np.minimum(rule[k*2+1], ranges[k*2+1]) > \
                        np.minimum(other[k*2+1], ranges[k*2+1])):
                    is_covered = False
                k += 1
            if is_covered:
                flag = True
                break
            j += 1
        if not flag:
            bool_1d[rule_len - 1 - i] = True
        i += 1
    bool_1d[0] = True
    return bool_1d


class RuleSet:
    def __init__(self, rules=None, rules_data=None):
        if rules is not None:
            self.rules_data = python_to_numpy(rules)
        elif rules_data is not None:
            self.rules_data = rules_data
        else:
            raise ValueError("One of rules or rules_data must be given")

        if len(self.rules_data) == 0:
            self.rules_hash = 0
        else:
            self.rules_hash = np.sum(self.rules_data)
            if (DEBUG_EQUALITY and
                  len(np.unique(self.rules_data, axis=0)) != len(self.rules_data)):
                raise ValueError("Duplicate rules found")

    def __len__(self):
        return len(self.rules_data)

    def __eq__(self, other):
        if len(self.rules_data) != len(other.rules_data):
            return False
        if self.rules_hash != other.rules_hash:
            return False
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

        start2 = time.time()
        numba_mask = prune_numba(self.rules_data, ranges)
        pruned_rules = self.rules_data[numba_mask]
        np_time = time.time() - start2

        if DEBUG:
            rules = numpy_to_python(self.rules_data)
            start = time.time()
            bool_1d = np.zeros(len(rules), dtype=bool)
            new_rules = []
            rule_len = len(rules)
            for i in range(rule_len - 1):
                rule = rules[rule_len - 1 - i]
                flag = False
                for j in range(rule_len - 1 - i):
                    if rule.is_covered_by(rules[j], ranges):
                        flag = True
                        break
                if not flag:
                    new_rules.append(rule)
                    bool_1d[rule_len - 1 - i] = True
            new_rules.append(rules[0])
            bool_1d[0] = True
            new_rules.reverse()
            py_time = time.time() - start

            print("prune speedup", py_time / np_time, len(self.rules_data))
            assert np.array_equal(bool_1d, numba_mask)

        return RuleSet(rules_data=pruned_rules)

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
