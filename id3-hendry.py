# coding=utf-8
import ast
import sys
import pandas as pd
from math import log2
from sklearn.preprocessing import LabelEncoder
import random
import os

positive_value = 'Yes'
negative_value = 'No'
unknown_value = '?'

class Node:
    def __init__(self):
        self.decision_attribute = ''
        self.test_branch = {}
        self.leaf_label = 'Node'


def print_tree(tree, depth=0, indent=4, requirement=None):
    if requirement is None:
        print("{}{}".format(" "*(indent*depth), tree.decision_attribute))
    else:
        if tree.leaf_label is 'Node':
            str_format = "{} == {} --> ({} ?)"
            print(str_format.format(" " * (indent*depth), requirement, tree.decision_attribute))

        else:
            str_format = "{} == {} --> {}"
            if tree.leaf_label:
                print(str_format.format(" " * (indent*depth),requirement, positive_value))
            else:
                print(str_format.format(" " * (indent*depth),requirement, negative_value))

    if tree.test_branch is not None:
        for req_path, child_node in tree.test_branch.items():
            print_tree(child_node, depth=depth+1, requirement=req_path)

# def print_tree(root, key_value_string, target_attribute):
#     if root.leaf_label != 'Node':
#         if root.leaf_label:
#             print(key_value_string + '\b\b\b\b' + ' → ' + '[' + target_attribute + ' = ' + positive_value + ']')
#         else:
#             print(key_value_string + '\b\b\b\b' + ' → ' + '[' + target_attribute + ' = ' + negative_value + ']')
#     if root.leaf_label == 'Node':
#         for v in root.test_branch.keys():
#             tmp = key_value_string + '[' + root.decision_attribute + ' = ' + str(v) + '] ' + ' ∧ '
#             print_tree(root.test_branch[v], tmp, target_attribute)
#     return


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


def information_gain(s, target_attribute, attribute, split=False):
    s_size = len(s)                            # |S|
    entropy_s = entropy(s, target_attribute)   # Entropy(S)
    values = s[attribute].unique()
    weighted_entropy_summary = 0
    split_entropy = 0
    
    for v in values:                           # ∑
        s_v = s[s[attribute] == v]             # Sv
        s_size_v = len(s_v)                    # |Sv|
        
        if split:
            # Calculate split entropy = - ∑ (|Sv| / |S|) log2 (|Sv| / |S|) 
            split_entropy += (-1.0) * ((s_size_v * 1.0 / s_size) * log2(s_size_v * 1.0 / s_size))
        
        entropy_s_v = entropy(s_v, target_attribute)
        weighted_entropy_summary += s_size_v * entropy_s_v / s_size

    gain = entropy_s - weighted_entropy_summary

    if split:
        # Return Gain Ratio
        try:
            gain_ratio = gain * 1.0 / split_entropy
            return gain_ratio
        except ZeroDivisionError:
            raise ZeroDivisionError("Error, split entropy cannot be zero")
        
    else:
        # Return Information Gain
        return gain


def id3_build_tree(examples, target_attribute, attributes):
    root = Node()
    if (examples[target_attribute] == positive_value).all():
        root.leaf_label = True
        return root
    if (examples[target_attribute] == negative_value).all():
        root.leaf_label = False
        return root
    if not attributes:
        root.leaf_label = examples[target_attribute].mode()[0] == positive_value
        return root
    ig = []
    for attribute in attributes:
        ig.append(information_gain(examples, target_attribute, attribute, True))
    a = attributes[ig.index(max(ig))]
    # print(a)
    root.decision_attribute = a
    values = examples[a].unique()
    for vi in values:
        examples_vi = examples[examples[a] == vi]
        if examples.empty:
            new_node = Node()
            new_node.leaf_label = examples[target_attribute].mode()[0] == positive_value
        else:
            new_node = id3_build_tree(examples_vi, target_attribute, [i for i in attributes if i != a])
        if vi != unknown_value:
            root.test_branch.update({vi: new_node})
    return root


def load_config(config_path):
    with open(config_path, 'r') as file:
        data = file.read().replace('\n', '')
    return ast.literal_eval(data)


if __name__ == "__main__":
    config_path = sys.argv[1]
    print("Config file path: {}\n".format(config_path))
    config = load_config(config_path)

    target_attribute = config['target_attribute']
    attributes = config['data_project_columns']
    df = pd.read_csv(config['data_file'])

    for drop_tuple in config['data_drop']:
        df = df.drop(drop_tuple[0], drop_tuple[1])

    id3_tree = id3_build_tree(df, target_attribute, attributes) 

    # print(id3_tree.decision_attribute, id3_tree.leaf_label)
    print_tree(id3_tree)
