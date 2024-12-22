import asyncio
import time
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import entropy
from collections import Counter
import plotly.graph_objects as go

class Async_Cache():
    def __init__(self, seconds_til_expire=120, interval=60) -> None:
        self.cache = {}
        self.seconds_til_expire = seconds_til_expire
        self.interval = interval
        asyncio.create_task(self.clean_up())

    async def set_key_value(self, key, value):
        self.cache[key] = [value, time.time()]

    async def get_value_from_key(self, key):
        if key in self.cache:
            value = self.cache[key]
            if time.time() - value[1] < self.seconds_til_expire:
                return value[0]
            else:
                del self.cache[key]
        return None

    async def clean_up(self):
        while True:
            await asyncio.sleep(self.interval)
            current_time = time.time()
            keys_to_delete = [key for key, (_, timestamp) in self.cache.items()
                                if current_time - timestamp >= self.seconds_til_expire]
            for key in keys_to_delete:
                del self.cache[key]

async def get_stock_data(cache, tickers, start, end):
    data = {}

    for ticker in tickers:
        key = f"{ticker}_{start}_{end}"
        cached_data = await cache.get_value_from_key(key)
        if cached_data is not None:
            data[ticker] = cached_data
            continue

        try:
            df = pd.read_csv(f'../CSVs/{ticker}_{start}_{end}_returns.csv', index_col=0, parse_dates=True)
            if cache:
                await cache.set_key_value(key, df)

            data[ticker] = df
        except Exception as e:
            df = yf.download(ticker, start=start, end=end)
            if cache:
                await cache.set_key_value(key, df)
            df.to_csv(f'../CSVs/{ticker}_{start}_{end}_returns.csv')
            data[ticker] = df

    return data

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        """
        Node class represents a single node in the decision tree.

        Parameters:
        feature: int, the index of the feature used to split at this node
        threshold: float, the threshold value to split the data
        left: Node, the left child node
        right: Node, the right child node
        value: int or None, the class label if this is a leaf node, None otherwise
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if the current node is a leaf node (i.e., it has no children)."""
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        """
        DecisionTree class implements a decision tree classifier.

        Parameters:
        min_samples_split: int, the minimum number of samples required to split a node
        max_depth: int, the maximum depth of the tree
        n_features: int or None, the number of features to consider when looking for the best split
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        Fit the decision tree to the training data.

        Parameters:
        X: array-like, shape (n_samples, n_features), feature matrix
        y: array-like, shape (n_samples,), target labels
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self.grow_tree(X, y)

    def grow_tree(self, X, y, depth=0):
        """
        Recursively grow the tree by splitting the data at each node.

        Parameters:
        X: array-like, shape (n_samples, n_features), feature matrix
        y: array-like, shape (n_samples,), target labels
        depth: int, the current depth of the tree
        """
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Stopping criteria: max depth, single class label, or not enough samples
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        # Randomly select a subset of features to consider for the split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best split based on information gain
        best_feature, best_thresh = self.best_split(X, y, feat_idxs)
        
        # Split the data into left and right subsets based on the best feature and threshold
        left_idxs, right_idxs = self.split(X[:, best_feature], best_thresh)
        left = self.grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_feature, best_thresh, left, right)

    def best_split(self, X, y, feat_idxs):
        """
        Find the best feature and threshold to split the data.

        Parameters:
        X: array-like, shape (n_samples, n_features), feature matrix
        y: array-like, shape (n_samples,), target labels
        feat_idxs: array-like, list of feature indices to consider for splitting

        Returns:
        best_feature: int, the feature index for the best split
        best_thresh: float, the threshold value for the best split
        """
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self.information_gain(y, X_column, threshold)
                
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def information_gain(self, y, X_column, threshold):
        """
        Calculate the information gain from a potential split.

        Parameters:
        y: array-like, shape (n_samples,), target labels
        X_column: array-like, shape (n_samples,), feature values for a particular feature
        threshold: float, the threshold value for the split

        Returns:
        information_gain: float, the information gain of the split
        """
        # Calculate the entropy of the parent node
        parent_entropy = self.entropy(y)

        # Generate the split and calculate the weighted average entropy of the children
        left_idxs, right_idxs = self.split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # Calculate the entropy of the child nodes
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self.entropy(y[left_idxs]), self.entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # Calculate the information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def split(self, X_column, split_thresh):
        """
        Split the data into left and right subsets based on a threshold.

        Parameters:
        X_column: array-like, shape (n_samples,), feature values for a particular feature
        split_thresh: float, the threshold value for the split

        Returns:
        left_idxs: array-like, indices of the samples in the left subset
        right_idxs: array-like, indices of the samples in the right subset
        """
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def entropy(self, y):
        """
        Calculate the entropy of a set of labels.

        Parameters:
        y: array-like, shape (n_samples,), target labels

        Returns:
        En: float, the entropy of the labels
        """
        base = 2
        _, counts = np.unique(y, return_counts=True)
        probs = counts / np.sum(counts)
        En = entropy(probs, base=base)
        # [1, 2, 2, 3, 4] -> [1, 2, 1, 1], 5 -> entropy([1/5, 2/5, 1/5, 1/5]) != entropy([1, 2, 2, 3, 4])

        return En

    def most_common_label(self, y):
        """
        Find the most common label in a set of labels.

        Parameters:
        y: array-like, shape (n_samples,), target labels

        Returns:
        most_common_label: int, the most common label
        """
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        """
        Predict the class labels for a set of samples.

        Parameters:
        X: array-like, shape (n_samples, n_features), feature matrix

        Returns:
        predictions: array-like, shape (n_samples,), predicted class labels
        """
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def traverse_tree(self, x, node):
        """
        Traverse the decision tree to make a prediction for a single sample.

        Parameters:
        x: array-like, shape (n_features,), a single sample
        node: Node, the current node in the decision tree

        Returns:
        value: int, the predicted class label
        """
        if node.is_leaf_node():
            return node.value

        # Recur to the left or right child based on the feature value
        if x[node.feature] <= node.threshold:
            return self.traverse_tree(x, node.left)
        return self.traverse_tree(x, node.right)
    
class GBDT:
    def __init__(self, n_estimators=5, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        # mean of y for regression
        self.y_mean = np.mean(y)
        y_pred = np.full_like(y, self.y_mean)
        
        for _ in range(self.n_estimators):
            # Compute residuals
            residuals = y - y_pred
            
            # Create and fit a decision tree to the residuals
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X, residuals)
            self.trees.append(tree)
            
            # Update predictions with the new tree
            update = self.learning_rate * tree.predict(X)
            y_pred += update

    def predict(self, X):
        # initial prediction
        y_pred = np.full(X.shape[0], self.y_mean)
        
        # Add predictions from each tree with learning rate
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        
        return y_pred

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

async def main():
    cache = Async_Cache()
    
    tickers = ["QQQ"]
    start_date = '2021-01-01'
    end_date = '2024-01-01'

    stock_data = await get_stock_data(cache, tickers, start_date, end_date)

    for ticker, df in stock_data.items():
        df = df.copy()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df = df.dropna()

        log_returns = df['log_returns'].values
        seq_length = 10
        X, y = create_sequences(log_returns, seq_length)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

        X_train = X_train.reshape((X_train.shape[0], -1))
        X_test = X_test.reshape((X_test.shape[0], -1))

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        gbdt = GBDT(max_depth=2)
        gbdt.fit(X_train, y_train)

        predictions = gbdt.predict(X_test)

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            y=y_test,
            mode='lines',
            name='Actual',
            line=dict(color='blue', width=2)
        ))

        fig.add_trace(go.Scatter(
            y=predictions.flatten(),
            mode='lines',
            name='Predicted',
            line=dict(color='orange', width=2)
        ))

        fig.update_layout(
            title=f"Gradient Boosted Decision Tree Predictions vs Actual for {ticker}",
            xaxis_title="Index",
            yaxis_title="Log Returns",
            legend=dict(x=0, y=1),
            template="plotly_dark"
        )

        fig.show()

if __name__ == "__main__":
    asyncio.run(main())