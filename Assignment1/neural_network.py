import time
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import ShuffleSplit
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from Assignment1.utility import read_data
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)


def neural_network(dataset):
    # read data & standardize
    x_train, y_train, x_test, y_test = read_data(dataset)

    # Method1: cross validation to find best value of H
    H = [i for i in range(1, 11)]
    acc = []
    for h in H:
        model = MLPClassifier(hidden_layer_sizes=h, batch_size=128, shuffle=False)
        rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        accuracy = []
        for train_index, validation_index in rs.split(x_train):
            train_set_X = x_train[train_index]
            train_set_Y = y_train[train_index]
            validation_set_X = x_train[validation_index]
            validation_set_Y = y_train[validation_index]
            model.fit(train_set_X, train_set_Y)
            accuracy.append(model.score(validation_set_X, validation_set_Y))
        print("H = {0}: Average Accuracy = {1}".format(h, np.average(accuracy)))
        acc.append(np.average(accuracy))
    best_h = acc.index(max(acc))
    print("==> Cross Validation: H = {0} got the max accuracy {1}".format(H[best_h], acc[best_h]))
    # visualize the cross validation
    label = dataset.split("/")[-1].replace('.npz', '')
    plt.plot(H, acc, 'o-', label=label)
    plt.xlim(0, 11)
    plt.ylim(0.5, 1.0)
    plt.legend(loc="best")
    plt.grid()
    plt.ylabel('Average Accuracy')
    plt.xlabel('Hidden Layer Size')

    # # Method2: Using model_selection.GridSearchCV for hyperparameter tuning
    # parameter = {'hidden_layer_sizes': H}
    # model = MLPClassifier(batch_size=128)
    # clf = GridSearchCV(model, parameter)
    # clf.fit(x_train, y_train)
    # print("==> GridSearchCV: {0} got the max accuracy {1}".format(clf.best_params_, clf.best_score_))

    # performance using the best value of H
    model = MLPClassifier(hidden_layer_sizes=10,
                          solver='lbfgs',
                          batch_size=128,
                          shuffle=False,
                          # alpha=0.1 // regularization term
                          )
    begin_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    print("Iteration = {0} times, Time = {1}".format(model.n_iter_, end_time - begin_time))
    print("Training Acc: {0}, Test Acc: {1}".format(model.score(x_train, y_train), model.score(x_test, y_test)))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, model.predict(x_test)))
    print("AUC:")
    # print(roc_auc_score(y_test, np.max(model.predict_proba(x_test), axis=1)))
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, model.predict(x_test))
    print(auc(false_positive_rate, true_positive_rate))
    # draw ROC curve
    # plt.figure()
    # plt.title("Receiver Operating Charascteristic(ROC)")
    # plt.plot(false_positive_rate, true_positive_rate)
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.show()
