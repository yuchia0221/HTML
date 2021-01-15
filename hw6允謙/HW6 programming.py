import numpy as np
from scipy.stats import mode
import pandas as pd
import time

# Gini_index definition


def impurity(y):
    mu_pos = np.count_nonzero(y == 1) / len(y)
    return 1 - mu_pos ** 2 - (1 - mu_pos) ** 2

# terminal criterion


def terminal(X: np.array, y: np.array) -> int:

    # If all the y is the same, then report y
    if len(np.unique(y)) == 1:
        return True

    # If all the x is the same, then report the most frequent y
    elif len(np.unique(X, axis=0)) == 1:
        print("All Xn are the same")
        return True
    else:
        return False


def branch_learning(X, y):
    # Combine X and y to df
    df = np.insert(X, X.shape[1], y, axis=1)
    bEin, btheta, feature = 99999, 0, 0
    leftX, rightX, lefty, righty = None, None, None, None

    # For every column
    for i in range(X.shape[1]):
        # The whole df sort by ith column
        ith_df = df[df[:, i].argsort()]
        x_ith = ith_df[:, i]
        theta = [(a + b) / 2 for a, b in zip(x_ith[:-1], x_ith[1:])]
        #print(x_ith, theta)

        for j in range(1, len(y)):
            y1, y2 = ith_df[:j, -1], ith_df[j:, -1]
            Ein = len(y1) * impurity(y1) + len(y2) * impurity(y2)
            if Ein < bEin:
                bEin = Ein
                btheta = theta[j - 1]
                feature = i
                leftX, rightX, lefty, righty = ith_df[:j,
                                                      :-1], ith_df[j:, :-1], y1, y2

    return feature, btheta, leftX, rightX, lefty, righty


class node():

    def __init__(self):
        self.feature = None
        self.threshhold = None
        self.left = None
        self.right = None
        self.value = None

    def train(self, X: np.array, y: np.array):
        if terminal(X, y):
            self.value = mode(y)[0][0]
        else:
            self.feature, self.threshhold, X1, X2, y1, y2 = branch_learning(
                X, y)
            self.left, self.right = node(), node()
            self.left.train(X1, y1)
            self.right.train(X2, y2)

    def predict(self, X: np.array, y: np.array):
        if self.value != None:
            return self.value
        else:
            if X[self.feature] < self.threshhold:
                return self.left.predict(X, y)
            else:
                return self.right.predict(X, y)


def problem14(train, test):
    print("Problem 14: ")
    trainX, trainy = train.drop([10], axis=1), train[10]
    testX, testy = test.drop([10], axis=1), test[10]

    DecisionTree = node()
    before = time.time()

    DecisionTree.train(np.array(trainX),
                       np.array(trainy))

    after = time.time()
    print(f"Training cost {after - before:.2f}")

    before = time.time()

    y_pred = [DecisionTree.predict(x, y) for x, y in zip(
        np.array(testX), np.array(testy))]

    after = time.time()
    print(f"Testing cost {after - before:.2f}")
    Eout = sum([y != y_hat for y, y_hat in zip(y_pred, testy)]) / len(y_pred)
    print(f"In pb14, Eout is {Eout:.2f}")
    print("=" * 40)

    return


def problem15(train, test):
    print("Problem 15: ")
    Eoutlist = []
    Forest = []

    before = time.time()
    for _ in range(20):
        train_15, test_15 = train.sample(
            500, replace=True), test

        trainX, trainy = train_15.drop([10], axis=1), train_15[10]
        testX, testy = test_15.drop([10], axis=1), test_15[10]

        DecisionTree = node()

        DecisionTree.train(np.array(trainX),
                           np.array(trainy))

        Forest.append(DecisionTree)

        y_pred = [DecisionTree.predict(x, y) for x, y in zip(
            np.array(testX), np.array(testy))]

        Eout = sum([y != y_hat for y, y_hat in zip(
            y_pred, testy)]) / len(y_pred)
        Eoutlist.append(Eout)

    after = time.time()
    print(f"Total cost {after - before:.2f}")

    avgEout = sum(Eoutlist) / 2000

    print(f"In pb15, average Eout is {avgEout:.2f}")
    print("=" * 40)

    return Forest


def problem16(train, Forest):
    print("Problem 16: ")
    G = []
    trainX, trainy = train.drop([10], axis=1), train[10]
    for tree in Forest:
        y_pred = [tree.predict(x, y) for x, y in zip(
            np.array(trainX), np.array(trainy))]
        G.append(y_pred)

    G = np.array(G)
    y_pred_G = [1 if i == 1 else -
                1 for i in np.apply_along_axis(sum, 0, G) > 0]
    Ein = sum([y != y_hat for y, y_hat in zip(
        y_pred_G, trainy)]) / len(y_pred_G)

    print(f"In pb16, Ein is {Ein:.2f}")
    print("=" * 40)


def problem17(test, Forest):
    print("Problem 17: ")
    G = []
    testX, testy = test.drop([10], axis=1), test[10]
    for tree in Forest:
        y_pred = [tree.predict(x, y) for x, y in zip(
            np.array(testX), np.array(testy))]
        G.append(y_pred)

    G = np.array(G)
    y_pred_G = [1 if i == 1 else -
                1 for i in np.apply_along_axis(sum, 0, G) > 0]
    Eout = sum([y != y_hat for y, y_hat in zip(
        y_pred_G, testy)]) / len(y_pred_G)

    print(f"In pb17, Eout is {Eout:.2f}")
    print("=" * 40)


if __name__ == "__main__":
    train = pd.read_csv("hw6_train.dat.txt", sep=" ", header=None)
    test = pd.read_csv("hw6_test.dat.txt", sep=" ", header=None)

    #train = train.sample(5)
    #test = test.sample(5)

    problem14(train, test)
    Forest = problem15(train, test)
    problem16(train, Forest)
    problem17(test, Forest)


print("end of ducument")
