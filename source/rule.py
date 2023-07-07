import numpy as np

class Rule:
    def __init__(self, class_=None, available_attibutes=None, coverage=0.0):
        self.attributes = available_attibutes
        self.class_ = class_
        self.p = 0
        self.t = coverage
        self.antecedent = {}

    def accuracy(self):
        if self.t <= 0:
            return -1
        else:
            return self.p / self.t

    def is_perfect(self):
        return self.accuracy() == 1.0

    def coverage(self):
        return self.t

    def not_used_attrs(self):
        return sorted(set(self.attributes) - set(list(self.antecedent.keys())))

    def insert_value(self, attr, value):
        self.antecedent[attr] = value

    def evaluate(self, instances, targets):
        covered = []
        corrects = []
        for inst, c in zip(instances, targets):
            if self._match(inst):
                covered.append(inst)
                if self.class_ == c:
                    corrects.append(inst)

        self.p, self.t = len(corrects), len(covered)

    def get_antecedent(self):
        return self.antecedent

    def get_class(self):
        return self.class_

    def extend(self, newRule):
        for key, val in newRule.get_antecedent().items():
            if key not in self.antecedent:
                self.antecedent[key] = val

    def inference(self, instances_ids, instances):
        newSet = []
        for id in instances_ids:
            if self.covered(instances[id]):
                newSet.append(id)
        return newSet

    def covered(self, instance):
        return self._match(instance)

    def precision_recall(self, X, Y):
        ant_cov, cls_cov = 0, 0
        for idx, item in enumerate(X):
            if self.covered(item):
                predicted_label = self.get_class()
                ant_cov += 1
                if Y[idx] == predicted_label:
                    cls_cov += 1

        precision = cls_cov/ant_cov
        recall = ant_cov/np.count_nonzero(Y == self.get_class())

        return precision, recall

    def print_rule(self, X, Y):
        prec, rec = self.precision_recall(X, Y)
        string = "IF "
        first = True
        for key, val in self.antecedent.items():
            if first:
                string += key + ' = ' + val
                first = False
            else:
                string += ' AND ' + key + ' = ' + val
        string += ' -> ' + self.class_ + '   (P=' + str(round(prec,2)) + ' R=' + str(round(rec,2)) + ')'
        return string

    def _match(self, instance):
        for key, val in self.antecedent.items():
            id = self.attributes.index(key)
            if val != instance[id]:
                return False
        return True
