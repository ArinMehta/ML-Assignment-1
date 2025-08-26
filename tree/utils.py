"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
from turtle import left
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

def check_ifreal(X: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_float_dtype(X)


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    df = X.copy()
    for c in df.columns:
        if not check_ifreal(df[c]):
            df = pd.get_dummies(df, columns=[c], drop_first=True)  # drop_first=True to avoid multicollinearity
    return df*1

    

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    bins = np.bincount(Y)
    probs = bins / Y.size
    return - np.sum([p * np.log2(p) for p in probs if p > 0])


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    try:
        bins = np.array(list(Counter(Y).values()))
        probs = bins / Y.size
        return 1 - np.sum([p**2 for p in probs])
    except:
        print(Y)
        raise ValueError("Error in Gini Index")

def gini_gain(parent: pd.Series, L: pd.Series, R: pd.Series) -> float:
    """
    Function to calculate Gini Gain.
    """
    weightL, weightR = L.size / parent.size, R.size / parent.size
    return gini_index(parent) - (weightL * gini_index(L) + weightR * gini_index(R))

def mse(Y: pd.Series) -> float:
    """
    Function to calculate Mean Squared Error.
    """
    return ((Y - Y.mean()) ** 2).mean()

def information_gain(parent: pd.Series, L: pd.Series, R: pd.Series) -> float:
    """
    Function to calculate Information Gain.
    """
    if check_ifreal(parent):
        return mse(parent) - (L.size / parent.size) * mse(L) - (R.size / parent.size) * mse(R)

    else:
        weightLeft, weightRight = L.size / parent.size, R.size / parent.size
        return entropy(parent) - (weightLeft * entropy(L) + weightRight * entropy(R))

def split(X: pd.DataFrame, Y: pd.Series, v: float, f: str) -> tuple:
    """
    Function to split the data.
    """
    tmp = X.copy()
    tmp["output"] = Y   # keep "output" as it is

    L = tmp[tmp[f] <= v]
    R = tmp[tmp[f] > v]

    Y_L, Y_R = L["output"], R["output"]
    X_L, X_R = L.drop(columns=["output"]), R.drop(columns=["output"])
    return X_L, X_R, Y_L, Y_R


def bestSplit(X: pd.DataFrame, Y: pd.Series, crit: str) -> tuple:
    """
    Function to find the best split.
    """
    bScore = -1
    bFeature, bThreshold = None, None

    for f in X.columns:
        uVals = np.sort(X[f].unique())  # sort unique values
        if len(uVals) <= 1:  # can't split if only one unique value
            continue
        
        splits = (uVals[1:] + uVals[:-1]) / 2  # candidate thresholds

        for t in splits:
            _, _, Y_L, Y_R = split(X, Y, t, f)
            if Y_L.empty or Y_R.empty:  # skip invalid splits
                continue

            if crit == "information_gain":
                cScore = information_gain(Y, Y_L, Y_R)
            else:
                cScore = gini_gain(Y, Y_L, Y_R)

            if cScore > bScore:
                bScore = cScore
                bFeature = f
                bThreshold = t
    
    return bFeature, bThreshold



# def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
#     """
#     Function to find the optimal attribute to split about.
#     If needed you can split this function into 2, one for discrete and one for real valued features.
#     You can also change the parameters of this function according to your implementation.

#     features: pd.Series is a list of all the attributes we have to split upon

#     return: attribute to split upon
#     """

#     # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).

#     pass


# def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
#     """
#     Funtion to split the data according to an attribute.
#     If needed you can split this function into 2, one for discrete and one for real valued features.
#     You can also change the parameters of this function according to your implementation.

#     attribute: attribute/feature to split upon
#     value: value of that attribute to split upon

#     return: splitted data(Input and output)
#     """

#     # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

#     pass
