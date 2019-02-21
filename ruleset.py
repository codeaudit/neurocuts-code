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
