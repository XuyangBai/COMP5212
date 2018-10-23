import numpy as np
import matplotlib.pyplot as plt
from Assignment1.logistic_regression import logistic_regression
from Assignment1.neural_network import neural_network
from Assignment1.svm import support_vector_machine_linear, support_vector_machine_RBF
import argparse

np.random.seed(0)  # To make sure every time the output is same

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='input the model number')
    parser.add_argument('-d', '--dataset', help="input the name of dataset")
    args = parser.parse_args()
    models = {
        "logistic_regression": logistic_regression,
        "neural_network": neural_network,
        "svm_rbf": support_vector_machine_RBF,
        "svm_linear": support_vector_machine_linear,
    }
    model = models[args.model]
    datasets = ["mouse", "pulsar_star", "wine"]
    dataset = args.dataset
    if dataset is None:
        for d in datasets:
            filename = './datasets/{}.npz'.format(d)
            model(filename)
        plt.show()
    else:
        filename = './datasets/{}.npz'.format(dataset)
        model(filename)
        plt.show()