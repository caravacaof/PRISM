PATH = './data/'
import matplotlib.pyplot as plt
import numpy as np


def read(name):
    # reading from the file
    file = open(PATH + name + ".data", "r")
    lines = file.readlines()
    attributes = []
    classes = []
    for l in lines:
        if name == 'balance-scale':
            attributes.append(l.rstrip().split(',')[1:])
            classes.append(l.rstrip().split(',')[0])
        elif name == 'nursery' or name == 'kr-vs-kp':
            attributes.append(l.rstrip().split(',')[:-1])
            classes.append(l.rstrip().split(',')[-1])
        elif name == 'lenses':
            attributes.append(l.split()[1:-1])
            classes.append(l.split()[-1])
        else:
            attributes.append(l.rstrip().split(',')[1:-1])
            classes.append(l.rstrip().split(',')[-1])

    return attributes, classes


def write_rules(path, rules, X, Y):
    with open(path, 'w') as f:
        for r in rules:
            f.write(r.print_rule(X, Y))
            f.write('\n')

def PR_plot(path, rules, X, Y):
    P, R = [], []
    for rule in rules:
        p, r = rule.precision_recall(X,Y)
        P.append(p)
        R.append(r)
    X_axis = np.arange(len(rules))
    if len(rules) > 120:
        plt.figure(figsize=(25, 8))
    elif len(rules) > 50:
        plt.figure(figsize=(20,8))
    else:
        plt.figure(figsize=(10,5))

    #plt.bar(X_axis - 0.2, P, 0.4, label='Precision')
    plt.bar(X_axis, R, label='Recall')

    plt.xticks(X_axis, X_axis)
    plt.xticks(rotation=90)
    plt.xlabel("Rule")
    plt.title("Rules recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)

