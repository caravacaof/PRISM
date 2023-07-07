import numpy as np
from source.rule import Rule


class PRISM:

    def __init__(self, examples, classes, attributes):
        self.attributes = attributes
        self.instances = np.array(examples)
        self.targets = np.array(classes)
        self.rules = []
        self.classes = list(np.unique(classes))

    def _possible_values(self, examples, attr):
        # returns possible values for a given attributes
        values = set()
        for inst in examples:
            i = self.attributes.index(attr)
            if self.instances[inst][i] not in values:
                values.add(self.instances[inst][i])
        return values

    def _get_best_rule_id(self, rules):
        max_id = 0
        for id, r in enumerate(rules):
            if r.accuracy() > rules[max_id].accuracy():
                max_id = id
            elif r.accuracy() == rules[max_id].accuracy() and r.coverage() > rules[max_id].coverage():
                max_id = id

        return max_id

    def _still_instances(self, class_, instances_id):
        targets = self.targets[instances_id]
        return np.where(targets == class_)[0].shape[0] > 0

    def fit(self):
        print('fitting model...')
        for c in self.classes:
            E = [i for i in range(self.instances.shape[0])]  # indexes of instances belonging the class
            while self._still_instances(c, E):
                rule = Rule(c, self.attributes, coverage=len(E))
                instances = E.copy()
                while not rule.is_perfect() and rule.coverage() > 1 and len(rule.not_used_attrs()) > 0:
                    rules = []
                    # for each pair attribute-value (A-V) not appearing in R
                    for attr in rule.not_used_attrs():
                        for val in self._possible_values(instances, attr):
                            Rav = Rule(c, self.attributes)
                            Rav.insert_value(attr, val)
                            Rav.evaluate(self.instances[instances], self.targets[instances])
                            rules.append(Rav)
                    best = rules[self._get_best_rule_id(rules)]
                    rule.extend(best)
                    instances = rule.inference(instances, self.instances)
                    rule.evaluate(self.instances[instances], self.targets[instances])
                # perfect rule
                self.rules.append(rule)
                E = self._remove_instances(E, rule)
        return self.rules

    def predict(self, X):
        print('predicting labels...')
        predicted_labels = len(X) * [None]
        for idx, item in enumerate(X):
            for rule in self.rules:
                if rule.covered(item):
                    predicted_labels[idx] = rule.get_class()
                    break
        return predicted_labels


    def _remove_instances(self, instances_ids, rule):
        newSet = []
        for id in instances_ids:
            if not rule.covered(self.instances[id]):
                newSet.append(id)
        return newSet

