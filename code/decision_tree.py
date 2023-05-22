import math
import pandas as pd

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute # attribute used for splitting
        self.label = label # label assigned to leaf node
        self.children = {} # dictionary of child nodes

class ID3DecisionTree:
    def __init__(self):
        self.tree = None # root node of decision tree

    def fit(self, X, y):
        # create pandas dataframe from input data and labels
        data = pd.DataFrame(X)
        data['label'] = y
        
        # build decision tree recursively
        self.tree = self._build_tree(data, data.columns[:-1])

    def predict(self, X):
        # make predictions for each instance in input data
        predictions = []
        for instance in X:
            node = self.tree
            while node.children:
                node = node.children[instance[node.attribute]]
            predictions.append(node.label)
        return predictions

    def _build_tree(self, data, attributes):
        # if all instances have the same label, return a leaf node
        if len(set(data['label'])) == 1:
            return Node(label=data['label'].iloc[0])
        
        # if no more attributes to split on, return a leaf node with the majority label
 
        if len(attributes) == 0:
            return Node(label=data['label'].value_counts().idxmax())

        # choose attribute with highest information gain to split on
        best_attr = self._choose_best_attribute(data, attributes)

        # create a new internal node with the chosen attribute
        node = Node(attribute=best_attr)

        # recursively build child nodes for each possible value of the chosen attribute
        for value in data[best_attr].unique():
            subset = data[data[best_attr] == value]
            if subset.empty:
                node.children[value] = Node(label=data['label'].value_counts().idxmax())
            else:
                node.children[value] = self._build_tree(subset, [attr for attr in attributes if attr != best_attr])

        return node

    def _choose_best_attribute(self, data, attributes):
        # calculate entropy of the current dataset
        entropy = self._calculate_entropy(data)

        # initialize variables to keep track of best attribute and its information gain
        best_attr = None
        max_info_gain = -math.inf

        # calculate information gain for each attribute and choose the one with the highest value
        for attr in attributes:
            info_gain = entropy - self._calculate_weighted_average_entropy(data, attr)
            if info_gain > max_info_gain:
                best_attr = attr
                max_info_gain = info_gain

        return best_attr

    def _calculate_entropy(self, data):
        # calculate entropy of the dataset
        labels = data['label'].value_counts()
        probs = labels / len(data)
        entropy = -sum(probs * probs.apply(math.log2))
        return entropy

    def _calculate_weighted_average_entropy(self, data, attribute):
        # calculate weighted average entropy of the dataset after splitting on the given attribute
        subsets = data.groupby(attribute)
        weighted_entropy = 0
        for value, subset in subsets:
            subset_entropy = self._calculate_entropy(subset)
            weight = len(subset) / len(data)
            weighted_entropy += weight * subset_entropy
        return weighted_entropy