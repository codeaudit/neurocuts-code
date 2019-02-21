class RuleSet:
    def __init__(self, rules):
        self.rules = rules

    def __len__(self):
        return len(self.rules)

    def get_rules(self):
        return self.rules

    def intersect(self, dimension, range_left, range_right):
        intersected_rules = []
        for rule in self.rules:
            if rule.is_intersect(dimension, range_left, range_right):
                intersected_rules.append(rule)

        return RuleSet(intersected_rules)
