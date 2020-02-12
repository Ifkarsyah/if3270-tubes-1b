# coding=utf-8
import pandas as pd
import numpy as np
from math import log2
from sklearn.preprocessing import LabelEncoder
import random

positive_value = 'Yes'
negative_value = 'No'
unknown_value = '?'

class Node:
    def __init__(self):
        self.value = ''
        self.children = {}
        self.label = 'Node'

def print_tree(root, depth=0, indent=4, requirement=None):
    if requirement is None:
        print("{}{}".format(" "*(indent*depth), root.value))
    else:
        
        if root.label is 'Node':
            str_format = "{} == {} --> ({} ?)"
            print(str_format.format(" " * (indent*depth), requirement, root.value))

        else:
            str_format = "{} == {} --> {}"
            if root.label:
                print(str_format.format(" " * (indent*depth),requirement, positive_value))
            else:
                print(str_format.format(" " * (indent*depth),requirement, negative_value))

    if root.children is not None:
        for req_path, child_node in root.children.items():
            print_tree(child_node, depth=depth+1, requirement=req_path)

def entropy(s, target_attribute):
    if s.empty:
        return 0
    ent = 0
    s_size = len(s)
    classes = s[target_attribute].unique()
    for i in classes:
        pi = (s[target_attribute] == i).sum()/s_size
        if pi != 0:
            ent += -pi * log2(pi)
    return ent

def information_gain(s, target_attribute, attribute):
    s_size = len(s)                            # |S|
    entropy_s = entropy(s, target_attribute)   # Entropy(S)
    values = s[attribute].unique()
    weighted_entropy_summary = 0
    for v in values:                           # ∑
        s_v = s[s[attribute] == v]             # Sv
        s_size_v = len(s_v)                    # |Sv|
        entropy_s_v = entropy(s_v, target_attribute)
        weighted_entropy_summary += s_size_v * entropy_s_v / s_size
    return entropy_s - weighted_entropy_summary

def id3_build_tree(examples, target_attribute, attributes):
    root = Node()

    if (examples[target_attribute] == positive_value).all():
        root.label = True
        return root
    
    if (examples[target_attribute] == negative_value).all():
        root.label = False
        return root
    
    if not attributes:
        root.label = examples[target_attribute].mode()[0] == positive_value
        return root
    ig = []
    for attribute in attributes:
        ig.append(information_gain(examples, target_attribute, attribute))
    a = attributes[ig.index(max(ig))]

    root.value = a
    values = examples[a].unique()
    for vi in values:
        examples_vi = examples[examples[a] == vi]
        if examples.empty:
            new_node = Node()
            new_node.label = examples[target_attribute].mode()[0] == positive_value
        else:
            new_node = id3_build_tree(examples_vi, target_attribute, [i for i in attributes if i != a])
        if vi != unknown_value:
            root.children.update({vi: new_node})
    return root

def id3_prune(examples, target_attribute, root, tree):
    if root.label != 'Node':
        return root
    for v in root.children.keys():
        root.children[v] = id3_prune(examples, target_attribute, root.children[v], tree)
    
    old_correctness = id3_correctness(tree, examples, target_attribute)
    root.label = examples[target_attribute].mode()[0] == positive_value
    new_correctness = id3_correctness(tree, examples, target_attribute)
    if new_correctness > old_correctness:
        return root
    else:
        root.label = 'Node'
        return root

def id3_classify(root, example):
    while root.label == 'Node':
        if example[root.value] in root.children:
            root = root.children[example[root.value]]
        elif example[root.value] == unknown_value:
            root = root.children[random.choice(list(root.children))]
        else:
            return 'Reject'
    if root.label:
        return positive_value
    else:
        return negative_value


def id3_correctness(root, example_test, target_attribute):
    test_size = len(example_test)
    correct = 0
    
    for i, row in example_test.iterrows():
        real_result = example_test.loc[i, target_attribute]
        id3_result = id3_classify(root, example_test.loc[i, :])
        
        if real_result == id3_result:
            correct += 1
    return correct/test_size
    
def split_data_set(data, fraction):
    threshold = int(fraction * len(data))
    return data.loc[np.random.choice(df.index, threshold)]

target_attribute = 'play'
attributes = ['outlook','temp','humidity','wind']
df = pd.read_csv("play_tennis.csv")
x = df.drop("day", 1)

data_pruning = split_data_set(x, 0.2)
id3_tree = id3_build_tree(x, target_attribute, attributes)
id3_tree = id3_prune(data_pruning, target_attribute, id3_tree, id3_tree)

print_tree(id3_tree)
