import math, sys
import datetime

from tree import *

class CutSplit(object):
    def __init__(self, rules):
        # hyperparameters
        self.leaf_threshold = 16    # number of rules in a leaf
        self.spfac = 4              # space estimation
        self.ip_threshold = 2**24      # t_sada

        # set up
        self.rules = rules

    def separate_rules(self, rules):
        def compute_ip_ranges(rule):
            return (rule.ranges[1] - rule.ranges[0],
                rule.ranges[3] - rule.ranges[2])

        def update_bins(rule, src_bins, dst_bins, bin_size):
            src_ip_range, dst_ip_range = compute_ip_ranges(rule)
            if src_ip_range < dst_ip_range:
                for i in range(rule.ranges[0] // bin_size,
                        math.ceil(rule.ranges[1] / bin_size)):
                    src_bins[i] += 1
            else:
                for i in range(rule.ranges[2] // bin_size,
                        math.ceil(rule.ranges[3] / bin_size)):
                    dst_bins[i] += 1

        # separate rules based on src and dst ip ranges
        # subset 0: small src ip, small dst ip
        # subset 1: small src ip, big dst ip
        # subset 2: big src ip, small dst ip
        # subset 3: big src ip, big dst ip
        rule_subsets = [[] for i in range(4)]
        bin_size = 2**12
        src_bins = [0 for i in range(2**20)]
        dst_bins = [0 for i in range(2**20)]
        for rule in rules:
            src_ip_range, dst_ip_range = compute_ip_ranges(rule)

            if src_ip_range <= self.ip_threshold and \
                    dst_ip_range <= self.ip_threshold:
                rule_subsets[0].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            elif src_ip_range <= self.ip_threshold and \
                    dst_ip_range > self.ip_threshold:
                rule_subsets[1].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            elif src_ip_range > self.ip_threshold and \
                    dst_ip_range <= self.ip_threshold:
                rule_subsets[2].append(rule)
                update_bins(rule, src_bins, dst_bins, bin_size)
            else:
                rule_subsets[3].append(rule)

        # add subset 0 to other subsets if it is too small
        if len(rule_subsets[0]) <= self.leaf_threshold:
            for rule in rule_subsets[0]:
                src_ip_range, dst_ip_range = compute_ip_ranges(rule)
                if src_ip_range < dst_ip_range:
                    rule_subsets[1].append(rule)
                else:
                    rule_subsets[2].append(rule)
            rule_subsets[0] = []

        # add subset 3 to subset 1 and subset 2
        for rule in rule_subsets[3]:
            src_sum = sum([1 for i in src_bins if i > self.leaf_threshold])
            dst_sum = sum([1 for i in dst_bins if i > self.leaf_threshold])
            if src_sum < dst_sum or \
                    (src_sum == dst_sum and \
                        len(rule_subsets[1]) <= len(rule_subsets[2])):
                rule_subsets[1].append(rule)
                for i in range(rule.ranges[0] // bin_size,
                        math.ceil(rule.ranges[1] / bin_size)):
                    src_bins[i] += 1
            else:
                rule_subsets[2].append(rule)
                for i in range(rule.ranges[2] // bin_size,
                        math.ceil(rule.ranges[3] / bin_size)):
                    dst_bins[i] += 1

        # sort rule by priority
        for rule_subset in rule_subsets:
            rule_subset.sort(key=lambda i: i.priority)

        return rule_subsets[0:3]

    def merge_rule_subsets(self, rule_subsets):
        return None

    def train(self):
        print(datetime.datetime.now(), "CutSplit starts")
        rule_subsets = self.separate_rules(self.rules)
        rule_subsets = self.merge_rule_subsets(rule_subsets)
        return
