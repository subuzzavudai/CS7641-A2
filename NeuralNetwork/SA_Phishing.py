"""
Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
find optimal weights to a neural network that is classifying abalone as having either fewer
or more than 15 rings.

Based on AbaloneTest.java by Hannah Lau
"""

from __future__ import with_statement

import os
import csv
import time
import sys


sys.path.append("ABAGAIL.jar")



from func.nn.backprop import BackPropagationNetworkFactory
from shared import SumOfSquaresError, DataSet, Instance
from opt.example import NeuralNetworkOptimizationProblem

import opt.SimulatedAnnealing as SimulatedAnnealing
import func.nn.activation.LogisticSigmoid as LogisticSigmoid

INPUT_LAYER = 30
HIDDEN_LAYER = 16
OUTPUT_LAYER = 1

# Back propagation Network Structure:
# classifier = MLPClassifier(activation='logistic', hidden_layer_sizes=(nodes), learning_rate='constant',
#                       max_iter=i, random_state=100, warm_start=False, momentum=momentum1, \
#                                    learning_rate_init=learning_rate1, solver='sgd')
# momentum1, learning_rate1 = 0.9, 0.25

def initialize_instances():
    """Read the abalone.txt CSV data into a list of instances."""
    train_instances = []
    test_instances = []

    data_test = os.path.join("..", "data", "phishing-test")
    data_train = os.path.join("..", "data", "phishing-train")

    # Read in the abalone.txt CSV file
    with open(data_train, "r") as phishing_train:
        reader = csv.reader(phishing_train)
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 0 else 1))
            train_instances.append(instance)

    with open(data_test, "r") as phishing_test:
        reader = csv.reader(phishing_test)
        for row in reader:
            instance = Instance([float(value) for value in row[:-1]])
            instance.setLabel(Instance(0 if float(row[-1]) == 0 else 1))
            test_instances.append(instance)

    return train_instances, test_instances


def train(oa, network, oaName, instances, measure, TRAINING_ITERATIONS):
    """Train a given network on a set of instances.

    :param OptimizationAlgorithm oa:
    :param BackPropagationNetwork network:
    :param str oaName:
    :param list[Instance] instances:
    :param AbstractErrorMeasure measure:
    """
    print "\nError results for %s\n---------------------------" % (oaName,)

    for iteration in xrange(TRAINING_ITERATIONS):
        oa.train()

        error = 0.00
        for instance in instances:
            network.setInputValues(instance.getData())
            network.run()

            output = instance.getLabel()
            output_values = network.getOutputValues()
            example = Instance(output_values, Instance(output_values.get(0)))
            error += measure.value(output, example)

        print "%0.03f" % error


def test(network, instances):
    correct = 0
    incorrect = 0

    for instance in instances:
        network.setInputValues(instance.getData())
        network.run()

        predicted = instance.getLabel().getContinuous()
        actual = network.getOutputValues().get(0)

        if abs(predicted - actual) < 0.5:
            correct += 1
        else:
            incorrect += 1

    return correct / float(correct + incorrect)


def main():
    optalgs = ['SA']

    OA = {
        'SA': SimulatedAnnealing
    }

    params = {
        'SA': [
            [1e2, 0.15], [1e2, 0.25], [1e2, 0.35], [1e2, 0.45], [1e2, 0.55],
            [1e2, 0.65], [1e2, 0.75], [1e2, 0.85], [1e2, 0.95]
        ]
    }

    identifier = {
        'SA': lambda p: str(p[1]).replace('.', '_')
    }

    iterations = [10, 50, 100, 200, 500, 1000, 2000, 3000, 4000, 5000]

    train_instances, test_instances = initialize_instances()

    data_set = DataSet(train_instances)
    factory = BackPropagationNetworkFactory()
    measure = SumOfSquaresError()

    for optalg in optalgs:
        for param in params[optalg]:
            output_filename = '%s-%s.csv' % (optalg, identifier[optalg](param))
            csv_file = open(output_filename, 'w')
            fields = ['num_iterations', 'train_accuracy', 'test_accuracy', 'train_time', 'test_time']
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            writer.writeheader()

            for num_iterations in iterations:
                network = factory.createClassificationNetwork([INPUT_LAYER, HIDDEN_LAYER, OUTPUT_LAYER], LogisticSigmoid())
                nnop = NeuralNetworkOptimizationProblem(data_set, network, measure)

                oa = OA[optalg](*(param + [nnop]))

                start = time.time()
                train(oa, network, optalg, train_instances, measure, num_iterations)
                end = time.time()
                train_time = end - start

                optimal_instance = oa.getOptimal()
                network.setWeights(optimal_instance.getData())
                train_accuracy = test(network, train_instances)

                start = time.time()
                test_accuracy = test(network, test_instances)
                end = time.time()
                test_time = end - start

                results = {
                    'num_iterations': num_iterations,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'train_time': train_time,
                    'test_time': test_time
                }

                print optalg, param, results
                writer.writerow(results)

            csv_file.close()
            print '------'

        print '***** ***** ***** ***** *****'


if __name__ == "__main__":
    main()

