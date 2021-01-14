import os
import numpy as np
import pandas as pd
from time import time
from typing import List, Iterable

if not os.path.isfile("train.csv") or not os.path.isfile("test.csv"):
    print("Downloading data...")
    train = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_train.dat", sep=r"\s", header=None)
    test = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_train.dat", sep=r"\s", header=None)

    train.to_csv("train.csv", index=False)
    test.to_csv("test.csv", index=False)
    print("Completed")
else:
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")


class TreeNode:
    def __init__(self, threshold, feature, value=None):
        self.left = None
        self.right = None
        self.value = value
        self.feature = feature
        self.threshold = threshold


def tree_predict(tree: TreeNode, data: Iterable) -> float:
    while (not tree.value):
        if float(data[tree.feature].values) >= tree.threshold:
            tree = tree.right
        else:
            tree = tree.left

    return tree.value


def learn_tree(data: pd.DataFrame) -> List[List]:
    if is_pure(data):
        print(data["10"])
        return TreeNode(None, None, data["10"].unique()[0])
    else:
        featureNum, threshold = find_best_feature(data)
        rightBranch = data[(data[featureNum] >= threshold)]
        leftBranch = data[(data[featureNum] < threshold)]
        DecisionTreeClassifier = TreeNode(threshold, featureNum, None)
        DecisionTreeClassifier.right = learn_tree(rightBranch)
        DecisionTreeClassifier.left = learn_tree(leftBranch)

        return DecisionTreeClassifier


def is_pure(data: pd.Series) -> bool:
    return len(pd.unique(data["10"])) == 1


def find_best_feature(data: pd.DataFrame) -> List:
    features = data.drop(["10"], axis=1).columns
    purityList = [find_best_threshold(data, feature) for feature in features]
    _, threshold, featureNumber = sorted(purityList, key=lambda x: (x[0], x[1], int(x[2])), reverse=False)[0]

    return [str(featureNumber), threshold]


def find_best_threshold(data: pd.DataFrame, feature: str) -> List[float]:
    featureList = []
    for threshold in generate_threshold(data[feature]):
        rightBran = data["10"][(data[feature] >= threshold)]
        leftBran = data["10"][(data[feature] < threshold)]
        weight = rightBran.count() / (rightBran.count() + leftBran.count())
        impurity = weight * gini_coef(rightBran) + (1 - weight) * gini_coef(leftBran)
        featureList.append([impurity, threshold, feature])

    return sorted(featureList, key=lambda x: (x[0], x[1], x[2]), reverse=False)[0]


def generate_threshold(data: pd.Series) -> List[float]:
    tmp = sorted(data.to_list())
    thresholdList = []
    for i in range(len(tmp)):
        if i == (len(tmp) - 1):
            break
        thresholdList.append((tmp[i] + tmp[i + 1]) / 2)

    return list(set(thresholdList))


def gini_coef(data: pd.Series) -> float:
    probability = (data.value_counts() / len(data)).values
    return 1 - np.sum(np.square(probability))


fakeData = pd.DataFrame({"1": [9, 8, 2, 2, 3], "2": [0, 2, 1, 3, 4], "10": [1, 1, 1, -1, -1]})
tree = learn_tree(fakeData)
tree_predict(tree, pd.DataFrame({"1": [9], "2": [0]}))
