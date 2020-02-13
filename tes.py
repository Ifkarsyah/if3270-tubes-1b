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
        self.children_label= {}
        self.label = 'Node'

def print_tree(root, depth=0, indent=4, requirement=None):
    if requirement is None:
        print("{}{}".format(" "*(indent*depth), root.value))
    else:
        
        if root.label is 'Node':
            str_format = "{} == {} / {} / {} --> ({} ?)"
            print(str_format.format(" " * (indent*depth), requirement, root.children_label, root.label, root.value))

        else:
            str_format = "{} == {} / {} / {} --> {}"
            if root.label:
                print(str_format.format(" " * (indent*depth), requirement, root.children_label, root.label, positive_value))
            else:
                print(str_format.format(" " * (indent*depth), requirement, root.children_label, root.label, negative_value))

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

    children_label = {}
    if (examples[target_attribute] == positive_value).all():
        root.label = True
        children_label[positive_value] = len(examples)
        children_label[negative_value] = 0
        root.children_label = children_label

        return root
    
    if (examples[target_attribute] == negative_value).all():
        root.label = False
        children_label[negative_value] = len(examples)
        children_label[positive_value] = 0
        root.children_label = children_label
        
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

    children_label[positive_value] = 0
    children_label[negative_value] = 0 
    
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
        
            children_label[positive_value] += new_node.children_label[positive_value]
            children_label[negative_value] += new_node.children_label[negative_value]

        if vi != unknown_value:
            root.children.update({vi: new_node})

    root.children_label = children_label
    return root

def id3_classify(root, example):
    while root.label == 'Node':
        if example[root.value] in root.children:
            root = root.children[example[root.value]]
        elif example[root.value] == unknown_value:
            root = root.children[random.choice(list(root.children))]
        else:
            return

    if root.label:
        return positive_value
    else:
        return negative_value


# def id3_correctness(root, example_test, target_attribute):
#     test_size = len(example_test)
#     correct = 0
    
#     for i, _ in example_test.iterrows():
#         real_result = example_test.loc[i, target_attribute]
#         print(example_test.loc[i, :])
#         print('+++++++++++++++++++++')
#         id3_result = id3_classify(root, example_test.loc[i, :])
        
#         # print(real_result, id3_result)

#         if real_result == id3_result:
#             correct += 1
#     return correct/test_size

def id3_error_rate(root):
    temp_min = 1E10
    temp_sum = 0
    
    for label in root.children_label:
        if (root.children_label[label] < temp_min):
            temp_min = root.children_label[label]
        
        temp_sum += root.children_label[label]

    return temp_min / temp_sum, temp_sum

def id3_prune(examples, target_attribute, root, tree):
    if root.label != 'Node':
        return root
    
    for val in root.children.keys():
        root.children[val] = id3_prune(examples, target_attribute, root.children[val], tree)
    
    error_rate_sub_tree, sum_sub_tree = id3_error_rate(root)

    # root.label = examples[target_attribute].mode()[0] == positive_value
    
    total_error_rate_node = 0
    for val in root.children.keys(): 
        error_rate_node, sum_node = id3_error_rate(root.children[val])
        total_error_rate_node += sum_node / sum_sub_tree * error_rate_node

    print(root.value, total_error_rate_node, error_rate_sub_tree)
    if total_error_rate_node < error_rate_sub_tree:
        return root
    else:
        root.label = 'Node'
        return root
    
def split_data_set(data, fraction):
    threshold = int(fraction * len(data))
    # return data.loc[np.random.choice(df.index, threshold)]

    return data.loc[0:threshold], data.loc[threshold+1:len(data)]

target_attribute = 'play'
attributes = ['outlook','temp','humidity','wind']
df = pd.read_csv("play_tennis.csv")
x = df.drop("day", 1)

data_pruning, data_example = split_data_set(x, 0.2)

id3_tree = id3_build_tree(data_example, target_attribute, attributes)
print_tree(id3_tree)

id3_tree = id3_prune(data_pruning, target_attribute, id3_tree, id3_tree)
print_tree(id3_tree)
