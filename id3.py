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

def entropy(data, target_attribute):
    ent = 0
    size = len(data)
    
    if data.empty:
        return 0
    
    classes = data[target_attribute].unique()
    for i in classes:
        p = (data[target_attribute] == i).sum() / size
        if p != 0:
            ent += -p * log2(p)
    
    return ent

def information_gain(data, target_attribute, attribute):
    size = len(data)
    entropy_s = entropy(data, target_attribute)
    
    single_attribute = data[attribute].unique()
    entropy_single = 0
    
    for attr in single_attribute:
        s_attr = data[data[attribute] == attr]

        entropy_temp = entropy(s_attr, target_attribute)
        entropy_single += len(s_attr) * entropy_temp / size
    
    return entropy_s - entropy_single

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
    
    A = attributes[ig.index(max(ig))]

    root.value = A
    values = examples[A].unique()

    for vi in values:
        examples_vi = examples[examples[A] == vi]
        
        if examples.empty:
            new_node = Node()
            new_node.label = examples[target_attribute].mode()[0] == positive_value
        
        else:
            next_attr = []
            for attr in attributes:
                if attr != A: 
                    next_attr.append(attr)

            new_node = id3_build_tree(examples_vi, target_attribute, next_attr)
        
        if vi != unknown_value:
            root.children.update({vi: new_node})

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

def id3_prune(examples, target_attribute, root, tree):
    if root.label != 'Node':
        return root
    
    for val in root.children.keys():
        root.children[val] = id3_prune(examples, target_attribute, root.children[val], tree)
    
    old_correctness = id3_correctness(tree, examples, target_attribute)
    root.label = examples[target_attribute].mode()[0] == positive_value
    new_correctness = id3_correctness(tree, examples, target_attribute)
    
    if new_correctness > old_correctness:
        return root
    else:
        root.label = 'Node'
        return root
    
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
