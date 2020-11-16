import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import log_loss, mean_squared_error, zero_one_loss

data = [[0, 1, -1], [1, -0.5, -1], [-1, 0, -1],
        [-1, 2, 1], [2, 0, 1], [1, -1.5, 1], [0, -2, 1]]

df = pd.DataFrame(data, columns=["x1", "x2", "y"])
df["x0"] = 1
df["x1^2"] = df["x1"].pow(2)
df["x2^2"] = df["x2"].pow(2)
df["x1*x2"] = df["x1"] * df["x2"]

df = df.reindex(columns=["x0", "x1", "x2", "x1^2", "x1*x2", "x2^2", "y"])
X, y = df.drop(["y"], axis=1).to_numpy(), df["y"].to_numpy()


def check_linear_separable():
    weightDict = {"a": np.array([-9, -1, 0, 2, -2, 3]), "b": np.array([-5, -1, 2, 3, -7, 2]), "c": np.array(
        [9, -1, 4, 2, -2, 3]), "d": np.array([2, 1, -4, -2, 7, -4]), "e": np.array([-7, 0, 0, 2, -2, 3])}
    for option, weight in weightDict.items():
        y_pred = np.dot(X, weight)
        y_pred = np.sign(y_pred)
        if zero_one_loss(y, y_pred) == 0:
            print(f"Answer is {option}")


if __name__ == "__main__":
    check_linear_separable()
