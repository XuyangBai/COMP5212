from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

from Assignment1.utility import read_data

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def support_vector_machine_linear(dataset):
    x_train, y_train, x_test, y_test = read_data(dataset)

    model = SVC(kernel='linear')
    begin_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    print("Time = {0}".format(end_time - begin_time))
    print("Training Acc: {0}, Test Acc: {1}".format(model.score(x_train, y_train), model.score(x_test, y_test)))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, model.predict(x_test)))


def support_vector_machine_RBF(dataset):
    x_train, y_train, x_test, y_test = read_data(dataset)

    # Using model_selection.GridSearchCV for hyperparameter tuning
    Gamma = [1, 0.1, 0.01, 0.001]
    parameters = {'gamma': Gamma}
    model = SVC(kernel='rbf')
    clf = GridSearchCV(model, parameters)
    clf.fit(x_train, y_train)
    # print("rank_test_score:",clf.cv_results_['rank_test_score']) # rank of test score.
    # print("mean_test_score:",clf.cv_results_['mean_test_score']) # mean test score.
    print("==> GridSearchCV: {0} got the max accuracy {1}".format(clf.best_params_, clf.best_score_))
    # visualization of cross validation
    label = dataset.split("/")[-1].replace('.npz', '')
    plt.plot(Gamma, clf.cv_results_['mean_test_score'], 'o-', label=label)
    plt.ylim(0.5, 1)
    plt.grid()
    plt.xscale('log')
    plt.legend(loc="best")
    plt.ylabel('Average Accuracy')
    plt.xlabel('Gamma')

    # performance on best model
    model = clf.best_estimator_
    begin_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()
    accuracy = model.score(x_test, y_test)
    print("Time = {0}".format(end_time - begin_time))
    print("Training Acc: {0}, Test Acc: {1}".format(model.score(x_train, y_train), accuracy))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, model.predict(x_test)))
    print("AUC:")
    print(roc_auc_score(y_test, model.predict(x_test)))