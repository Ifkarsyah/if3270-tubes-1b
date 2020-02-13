# coding=utf-8
import pandas as pd
from math import log2
from sklearn.preprocessing import LabelEncoder
import random

df = pd.read_csv("iris-full.csv")
target_attribute = list(df.columns)[-1]
goal_values = df[target_attribute].unique()

def get_attributes_and_split(s, attr_name):
    attributes_and_split = []
    values = list(set(df[attr_name]))
    for i in range(len(values) - 1):
        split_point = round((values[i] + values[i+1]) / 2, 2)
        attributes_and_split.append((attr_name, split_point))
    return attributes_and_split

attributes_name = list(df.columns)[:-1]
attributes_and_split = []
for attr in attributes_name:
    attributes_and_split.extend(get_attributes_and_split(df, attr))

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
            print(str_format.format(" " * (indent*depth),
                                    requirement, tree.decision_attribute))

        else:
            str_format = "{} == {} --> {}"
            for gv in goal_values:
                if tree.leaf_label == gv:
                    print(str_format.format(" " * (indent*depth),
                                            requirement, gv))

    if tree.test_branch is not None and tree.leaf_label is 'Node':
        for req_path, child_node in tree.test_branch.items():
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

    weighted_entropy_summary = 0

    if isinstance(attribute, str):
        values = s[attribute].unique()
        for v in values:                           # ∑
            s_v = s[s[attribute] == v]             # Sv
            s_size_v = len(s_v)                    # |Sv|
            entropy_s_v = entropy(s_v, target_attribute)
            weighted_entropy_summary += s_size_v * entropy_s_v / s_size
    else: # attribute = (sepal, 2.5)
        attr_name, split_point = attribute

        s_v = s[s[attr_name] <= split_point]
        entropy_s_v = entropy(s_v, target_attribute)
        if len(s) != 0:
            weighted_entropy_summary += len(s_v) * entropy_s_v / len(s)

        s_v = s[s[attr_name] > split_point]
        entropy_s_v = entropy(s_v, target_attribute)
        if len(s) != 0:
            weighted_entropy_summary += len(s_v) * entropy_s_v / len(s)

    return entropy_s - weighted_entropy_summary

def id3_build_tree(examples, target_attribute, attributes):
    root = Node()

    for gv in goal_values:
        if (examples[target_attribute] == gv).all():
            root.leaf_label = gv

    if len(attributes) == 0:
        mode = examples[target_attribute].mode()
        if len(mode) > 0:
            root.leaf_label = examples[target_attribute].mode()[0]
        return root
    else:
        ig = []
        for attribute in attributes:
            ig.append(information_gain(examples, target_attribute, attribute))
        a = attributes[ig.index(max(ig))]

        root.decision_attribute = a
        if isinstance(a, str): # discrete
            values = examples[a].unique()
            for vi in values:
                examples_vi = examples[examples[root.decision_attribute] == vi]
                new_node = id3_build_tree(examples_vi, target_attribute, [i for i in attributes if i != a])
                root.test_branch.update({vi: new_node})
        else: # continu --> a = (variable_name, split_point)
            attr_name, split_point = a
            # <=
            examples_vi = examples[examples[attr_name] <= split_point]
            if examples.empty:
                new_node = Node()
                new_node.leaf_label = examples[target_attribute].mode()
            else:
                new_node = id3_build_tree(examples_vi, target_attribute, 
                [i for i in attributes if (i != a and i[1] <= split_point)]
                )
                root.test_branch.update({'<=' + str(split_point): new_node})
            # >
            examples_vi = examples[examples[attr_name] > split_point]
            if examples.empty:
                new_node = Node()
                new_node.leaf_label = examples[target_attribute].mode()
            else:
                new_node = id3_build_tree(examples_vi, target_attribute, 
                [i for i in attributes if (i != a and i[1] > split_point)]
                )
                root.test_branch.update({'>' + str(split_point): new_node})

        return root



id3_tree = id3_build_tree(df, target_attribute, attributes_and_split)
print_tree(id3_tree)
