import numpy as np
from source.prism import PRISM
from source.utils import read, write_rules, PR_plot
import sys
import time
import pandas as pd

attribute_names = {
    'balance-scale': ['Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'],
    'nursery': ['parents', 'has_nurs', 'form', 'children', 'housing', 'finance', 'social'],
    'hayes-roth': ['hobby', 'age', 'educational_level', 'marital_status'],
    'kr-vs-kp': [
        "bkblk", "bknwy", "bkon8", "bkona", "bkspr", "bkxbq", "bkxcr", "bkxwp", "blxwp", "bxqsq", "cntxt", "dsopp",
        "dwipd", "hdchk", "katri", "mulch", "qxmsq", "r2ar8", "reskd", "reskr", "rimmx", "rkxwp", "rxmsq", "simpl",
        "skach", "skewr", "skrxp", "spcop", "stlmt", "thrsk", "wkcti", "wkna8", "wknck", "wkovl", "wkpos", "wtoeg"
    ],
    'lenses': ['age', 'visual', 'astig', 'tear']
}

if __name__ == '__main__':

    # default parameters
    dataset = 'hayes-roth'
    num_exec = 1
    save_dir = None

    # arguments reading
    args = sys.argv
    for flag, arg in zip(args, args[1:]):
        if flag == '-d':
            dataset = arg
        elif flag == '-n':
            num_exec = int(arg)
        elif flag == '-s':
            save_dir = arg

    # read dataset
    X, Y = read(dataset)

    # start executions
    accuracies = []
    coverages = []
    runtimes = []
    n_rules = []
    for i in range(num_exec):
        print('- Starting execution ' + str(i))
        # split training and test data
        mask = np.random.rand(len(X)) <= 0.8
        X = np.array(X)
        Y = np.array(Y)
        training_X = X[mask]
        testing_X = X[~mask]
        training_Y = Y[mask]
        testing_Y = Y[~mask]

        # Train model
        prism = PRISM(training_X, training_Y, attribute_names[dataset])
        start_time = time.time()
        rules = prism.fit()
        runtime = time.time() - start_time
        runtimes.append(runtime)
        n_rules.append(len(rules))
        print('- Number of rules: ' + str(len(rules)))
        write_rules('results/' + dataset + str(i) + '.rules', rules, training_X, training_Y)
        PR_plot('results/' + dataset + str(i) + '.png', rules, training_X, training_Y)
        print('- ' + dataset + str(i) + '.rules saved succesfully.')

        # Test model
        prediction = prism.predict(testing_X)
        accuracy = sum(1 for idx in range(len(testing_Y)) if testing_Y[idx] == prediction[idx]) / len(testing_Y)
        accuracies.append(accuracy)
        coverage = sum(1 for item in prediction if prediction is not None) / len(testing_Y)
        coverages.append(coverage)
        print('- Accuracy: ' + str(accuracy))
        print('- Coverage: ' + str(coverage))
        print('- Runtime: ' + str(round(runtime, 3)))
        print('##################################################################################################')

    # final results
    if not save_dir is None:
        results = pd.DataFrame([],
                               columns=['NumRules', 'Accuracy', 'Coverage', 'Runtime'])
        results['NumRules'] = n_rules
        results['Accuracy'] = accuracies
        results['Coverage'] = coverages
        results['Runtime'] = runtimes
        results.to_excel(save_dir + dataset + '_' + str(num_exec) + '.xlsx')

    print()
    print('AVERAGE RESULTS:')
    print()
    print('NUMBER OF RULES: ' + str(np.average(n_rules)))
    print('ACCURACY: ' + str(np.average(accuracies)))
    print('COVERAGE: ' + str(np.average(coverages)))
    print('RUNTIME: ' + str(round(np.average(runtimes), 3)))
