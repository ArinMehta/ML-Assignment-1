import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from tree.base import DecisionTree
from metrics import accuracy, precision, recall

np.random.seed(42)


# Generate dataset

X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2,
    random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Plot dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.show()


# Train/Test Split (single model on full features)

xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.3, random_state=42)
xtrain, xtest = pd.DataFrame(xtrain), pd.DataFrame(xtest)
ytrain, ytest = pd.Series(ytrain), pd.Series(ytest)

tree = DecisionTree(criterion="information_gain", maxDepth=3)
tree.fit(xtrain, ytrain)
y_hat = tree.predict(xtest)

print("Accuracy of our Decision Tree is " + str(accuracy(ytest, y_hat) * 100) + "%")
for cls in np.unique(ytest):
    print(f"Class {cls}:")
    print(f"  Precision: {precision(y_hat, ytest, cls):.4f}")
    print(f"  Recall:    {recall(y_hat, ytest, cls):.4f}")


# Manual K-Fold Cross Validation

n_folds = 5
dataset_size = len(X)
indices = np.arange(dataset_size)
np.random.shuffle(indices)

fold_sizes = np.full(n_folds, dataset_size // n_folds)
fold_sizes[:dataset_size % n_folds] += 1

current = 0
folds = []
for fold_size in fold_sizes:
    start, stop = current, current + fold_size
    folds.append(indices[start:stop])
    current = stop

fold_accuracies = []
print("\nK-Fold Validation Results:")
for i in range(n_folds):
    train_indices = np.concatenate([folds[j] for j in range(n_folds) if j != i])
    val_indices = folds[i]

    X_train, y_train = pd.DataFrame(X[train_indices]), pd.Series(y[train_indices])
    X_val, y_val = pd.DataFrame(X[val_indices]), pd.Series(y[val_indices])

    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_val)
    acc = accuracy(y_val, y_pred)
    fold_accuracies.append(acc)
    print(f"Fold {i + 1}, Accuracy: {acc:.4f}")

print(f"Mean accuracy is {np.mean(fold_accuracies):.4f}")


# Nested Cross Validation

def splitfold(x, n):
    i = np.arange(len(x))
    np.random.shuffle(i)
    return np.array_split(i, n)

def nestedvalid(X, y, depths=[2, 3, 4, 5, 6], ofold=5, ifold=5):
    oscore = []
    best_depths = []
    ofold_indices = splitfold(X, ofold)

    for i in range(ofold):
        test_indices = ofold_indices[i]
        train_indices = np.concatenate([ofold_indices[j] for j in range(ofold) if j != i])

        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        inner_folds_indices = splitfold(X_train, ifold)
        best_depth, best_score = None, -np.inf

        # Inner CV for depth selection
        for depth in depths:
            iscore = []
            for j in range(ifold):
                val_indices = inner_folds_indices[j]
                inner_train_indices = np.concatenate([inner_folds_indices[k] for k in range(ifold) if k != j])

                X_inner_train, X_inner_val = X_train[inner_train_indices], X_train[val_indices]
                y_inner_train, y_inner_val = y_train[inner_train_indices], y_train[val_indices]

                model = DecisionTree(criterion="information_gain", maxDepth=depth)
                model.fit(pd.DataFrame(X_inner_train), pd.Series(y_inner_train))
                acc = accuracy(y_inner_val, model.predict(pd.DataFrame(X_inner_val)))
                iscore.append(acc)

            mean_score = np.mean(iscore)
            if mean_score > best_score:
                best_score, best_depth = mean_score, depth

        best_depths.append(best_depth)
        print(f"Outer Fold {i+1}: Best depth = {best_depth}, Inner CV Mean Acc = {best_score:.4f}")

        # Train final model with best depth
        final_model = DecisionTree(criterion="information_gain", maxDepth=best_depth)
        final_model.fit(pd.DataFrame(X_train), pd.Series(y_train))
        y_pred = final_model.predict(pd.DataFrame(X_test))
        outer_acc = accuracy(y_test, y_pred)
        oscore.append(outer_acc)
        print(f"   Outer Fold {i+1} Test Accuracy = {outer_acc:.4f}\n")

    return np.mean(oscore) * 100, np.std(oscore), best_depths

mscore, stdcore, depths_chosen = nestedvalid(X, y, depths=[2, 3, 4, 5, 6], ofold=5, ifold=5)
print(f"\nNested CV: {mscore:.2f}% Accuracy")
print(f"Standard Deviation: {stdcore:.4f}")
print(f"Depths chosen per outer fold: {depths_chosen}")
print(f"Final Best Depth (most frequent): {max(set(depths_chosen), key=depths_chosen.count)}")
