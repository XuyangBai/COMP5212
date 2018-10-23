import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from Assignment1.utility import read_data


class LogisticRegression:
    def __init__(self, learning_rate_init=0.01, learning_rate_decay=None, reg=1):
        self.learning_rate_init = learning_rate_init
        self.lr_decay = learning_rate_decay
        self.regularization = reg
        self.W = None
        self.b = None

    def fit(self, x_train, y_train, max_iter=200, verbose=False, learning_curve=False):
        n_sample = x_train.shape[0]
        n_features = x_train.shape[1]
        y_train = y_train.reshape(n_sample, 1)
        self.W = np.random.random([n_features, 1])
        self.b = np.random.random()
        self.history = {'epoch': [], 'loss': [], 'training accuracy': []}

        for i in range(max_iter):
            y_pred = self.forward(x_train)
            loss = self.loss(y_train, y_pred)
            dW, db = self.backward(x_train, y_train, y_pred)
            if self.lr_decay is not None:
                self.lr = self.learning_rate_init / (1 + i * self.lr_decay)
            self.W -= dW.reshape(n_features, 1) * self.lr
            self.b -= db * self.lr

            # reocrd and visualize the change in performance during training.
            if i % 10 == 0 and verbose:
                accuracy = np.mean(self.predict(x_train) == y_train)
                self.history['epoch'].append(i)
                self.history['loss'].append(loss)
                self.history['training accuracy'].append(accuracy)
                print("Epoch {0}: Training Accuracy = {1}, Loss = {2}".format(i, accuracy, loss))

        if learning_curve:
            plt.figure()
            plt.subplot(2, 1, 1)
            plt.plot(self.history['epoch'], self.history['loss'], 'o-')
            plt.xlim(min(self.history['epoch']), max(self.history['epoch']) + 10)
            plt.ylim(min(self.history['loss']) * 0.9, max(self.history['loss']) * 1.1)
            plt.ylabel('Loss')
            plt.grid()
            plt.subplot(2, 1, 2)
            plt.plot(self.history['epoch'], self.history['training accuracy'], 'or-')
            plt.xlabel('Epoch')
            plt.ylabel('Training Accuracy')
            plt.ylim(min(self.history['training accuracy']) * 0.9, max(self.history['training accuracy']) * 1.1)
            plt.xlim(min(self.history['epoch']), max(self.history['epoch']) + 10)
            plt.ylim(0.4, 1)
            plt.grid()

    def forward(self, x_train):
        # x_train is [n_samples, n_features]
        # self.W is [n_features, 1]
        return self._sigmoid(x_train.dot(self.W) + self.b)

    def loss(self, y_true, y_pred):
        # return -1.0 / y_true.shape[0] * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(y_pred))
        return -1.0 / y_true.shape[0] * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(y_pred)) + \
               self.regularization * np.sum(self.W)

    def backward(self, x_train, y_train, y_pred):
        dW = -1 / y_train.shape[0] * np.sum((y_train - y_pred) * x_train, axis=0)
        db = -1 / y_train.shape[0] * np.sum(y_train - y_pred)
        return dW, db

    def predict(self, x):
        y = self.forward(x) > 0.5
        return y.astype(np.int)

    def _sigmoid(self, x):
        return 1.0 / (1 + np.exp(-x))


def logistic_regression(dataset):
    x_train, y_train, x_test, y_test = read_data(dataset)
    model = LogisticRegression(learning_rate_init=0.5, learning_rate_decay=0.05)
    begin_time = time.time()
    model.fit(x_train, y_train, max_iter=300, verbose=True, learning_curve=True)
    end_time = time.time()
    print("Time = {0}".format(end_time - begin_time))
    training_accuracy = np.mean(y_train == model.predict(x_train))
    test_accuracy = np.mean(y_test == model.predict(x_test))
    print("Training Acc: {0}, Test Acc: {1}".format(training_accuracy, test_accuracy))
    print('Confusion matrix: ')
    print(confusion_matrix(y_test, model.predict(x_test)))
    print("AUC:")
    print(roc_auc_score(y_test, model.forward(x_test)))
    plt.show()

# dataset = "../datasets/pulsar_star.npz"
# logestic_regression(dataset)
