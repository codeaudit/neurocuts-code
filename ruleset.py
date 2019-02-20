class RuleSet:
    def __init__(self, rules):
        self.rules = rules

    def intersect(self, dimension, range_left, range_right):
        child_rules = []
        for rule in self.rules:
            if rule.is_intersect(dimension, range_left, range_right):
                child_rules.append(rule)
        return RuleSet(child_rules)

    def get_rules(self):
        return self.rules

    def length(self):
        return len(self.rules)
