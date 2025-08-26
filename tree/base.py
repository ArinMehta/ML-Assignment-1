"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .utils import *


class Node:
    def __init__(self, feature, threshold, left, right, value) -> None:
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value


class DecisionTree:
    def __init__(self, criterion, maxDepth=5) -> None:
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.isRealOutput = False
        self.tree = None

    def fit(self, X, y) -> None:
        X = pd.DataFrame(X)
        y = pd.Series(y)

        # One-hot encode categorical variables
        X_encoded = one_hot_encoding(X)

        # Decide regression or classification tree
        self.isRealOutput = check_ifreal(y)
    
        # Build the decision tree
        self.tree = self.buildTree(X_encoded, y, 0)

        

    def buildTree(self, X, y, depth) -> Node:
        node = Node(None, None, None, None, None)

        # Stopping criteria
        if depth >= self.maxDepth or len(y.unique()) == 1:
            node.value = y.mean() if self.isRealOutput else y.mode()[0]
            return node

        # Find best split
        bestFeature, bestThreshold = bestSplit(X, y, self.criterion)

        # If no valid split found → leaf node
        if bestFeature is None:
            node.value = y.mean() if self.isRealOutput else y.mode()[0]
            return node

        # Split dataset
        leftX, rightX, leftY, rightY = split(X, y, bestThreshold, bestFeature)

        # Assign children
        node.feature = bestFeature
        node.threshold = bestThreshold
        node.left = self.buildTree(leftX, leftY, depth + 1)
        node.right = self.buildTree(rightX, rightY, depth + 1)

        return node

    def predictOne(self, node, row) -> float:
        if node.value is not None:
            return node.value
        if row[node.feature] <= node.threshold:
            return self.predictOne(node.left, row)
        return self.predictOne(node.right, row)

    def predict(self, X) -> np.ndarray:
        X = pd.DataFrame(X)
        X_encoded = one_hot_encoding(X)

        predictions = [self.predictOne(self.tree, X_encoded.iloc[i])
                       for i in range(X_encoded.shape[0])]
        return np.array(predictions)

    def plot(self) -> None:
     #         Function to plot the tree

     #         Output Example:
     #         ?(X1 > 4)
     #             Y: ?(X2 > 7)
     #                 Y: Class A
     #             N: Class B
     #         N: Class C
     #     Where Y => Yes and N => No
        def plot_node(node, depth):
            indent = "    " * depth
            if node.value is not None:
                print(f"{indent}Class {node.value}")
                return

            print(f"{indent}?({node.feature} > {node.threshold})")
            print(f"{indent}Y: ", end="")
            plot_node(node.right, depth + 1)
            print(f"{indent}N: ", end="")
            plot_node(node.left, depth + 1)

        plot_node(self.tree, 0)

    def plotGraph(self) -> None:
        def plot_node(node, depth, x, y, dx):
            if node is None:
                return

            label = (f'{node.feature}\n<= {node.threshold}'
                     if node.value is None else f'{node.value:.2f}')

            plt.text(x, y, label,
                     ha='center', va='center',
                     bbox=dict(facecolor='white',
                               edgecolor='black',
                               boxstyle='round,pad=0.5'))

            if node.left is not None:
                plt.plot([x, x - dx], [y, y - 1], 'k-')
                plot_node(node.left, depth + 1, x - dx, y - 1, dx / 2)

            if node.right is not None:
                plt.plot([x, x + dx], [y, y - 1], 'k-')
                plot_node(node.right, depth + 1, x + dx, y - 1, dx / 2)

        plt.figure(figsize=(12, 8))
        plot_node(self.tree, 0, 0, 0, 1)
        plt.axis('off')
        plt.show()


# @dataclass
# class DecisionTree:
#     criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
#     max_depth: int  # The maximum depth the tree can grow to

#     def __init__(self, criterion, max_depth=5):
#         self.criterion = criterion
#         self.max_depth = max_depth

#     def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
#         """
#         Function to train and construct the decision tree
#         """

#         # If you wish your code can have cases for different types of input and output data (discrete, real)
#         # Use the functions from utils.py to find the optimal attribute to split upon and then construct the tree accordingly.
#         # You may(according to your implemetation) need to call functions recursively to construct the tree. 

#         pass

#     def predict(self, X: pd.DataFrame) -> pd.Series:
#         """
#         Funtion to run the decision tree on test inputs
#         """

#         # Traverse the tree you constructed to return the predicted values for the given test inputs.

#         pass

#     def plot(self) -> None:
#         """

#         """
#         pass
