import os
import numpy as np
import pandas as pd
from typing import List

if not os.path.isfile("train.csv") or not os.path.isfile("test.csv"):
    print("Downloading data...")
    train = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_train.dat", sep="\s", header=None)
    test = pd.read_csv("https://www.csie.ntu.edu.tw/~htlin/course/ml20fall/hw6/hw6_train.dat", sep="\s", header=None)

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


def learn_tree(data: pd.DataFrame) -> List[List]:
    if is_pure(data):
        return DecisionTreeClassifier(None, None, data["10"][0])
    else:
        featureNum, threshold = find_best_feature(data)
        rightBranch = data[(data[featureNum] >= threshold)]
        leftBranch = data[~(data[featureNum] >= threshold)]

        DecisionTreeClassifier = TreeNode(threshold, featureNum, None)
        DecisionTreeClassifier.right = learn_tree(rightBranch)
        DecisionTreeClassifier.left = learn_tree(leftBranch)

        return DecisionTreeClassifier


def is_pure(data: pd.Series) -> bool:
    return len(pd.unique(data)) == 1


def find_best_feature(data: pd.DataFrame) -> List:
    features = data.columns
    purityList = [find_best_threshold(data, feature) for feature in features]
    bestpurity, threshold, featureNumber = sorted(purityList, key=lambda x: (x[0], x[1], x[2]), reverse=False)[0]

    return [str(featureNumber), threshold]


def find_best_threshold(data: pd.DataFrame, feature: str) -> List[float]:
    featureList = []
    for threshold in generate_threshold(data[feature]):
        rightBran = data["10"][(x >= threshold)]
        leftBran = data["10"][(x < threshold)]
        weight = rightBran.count() / (rightBran.size() + leftBran.count())
        purity = weight * gini(rightBran) - (1 - weight) * gini(leftBran)
        featureList.append([purity, threshold, feature])

    return sorted(featureList, key=lambda x: (x[0], x[1], x[2]), reverse=True)[0]


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


fakeData = pd.DataFrame({"A": [9, 8, 3, 2, 3], "B": [0, 0, 1, 3, 4], "10": [1, 1, 1, -1, -1]})
print(learn_tree(train))
