# coding=utf-8
from math import log2
import ast
import sys
import random
import os
import pandas as pd
import numpy as np

class Node:
    def __init__(self):
        self.value = '' # Menyimpan atribut yang akan menjadi child node
        self.children = {} # Menyimpan cabang DTL dengan ini sebagai root node
        self.children_label = {} # Menyimpan data frekuensi target dari root node
        self.label = 'Node' # Menyimpan label dari daun node

'''
Mencetak pohon secara rekursif
Pohon dengan bentuk
    a
   / \
  x   y
 /     \
b       c

akan dicetak menjadi bentuk
a
    == x --> b
    == y --> c 
'''
def print_tree(root, goal_values, depth=0, indent=4, requirement=None):
    if requirement is None:
        print("{}{}".format(" " * (indent * depth), root.value))
    
    else:    
        if root.label is 'Node':
            str_format = "{} == {} --> ({} ?)"
            print(str_format.format(" " * (indent * depth), requirement, root.value))

        else:
            str_format = "{} == {} --> {}"
            for gv in goal_values:
                if root.label == gv:
                    print(str_format.format(" " * (indent*depth), requirement, gv))

    if root.children is not None and root.label is 'Node':
        for req_path, child_node in root.children.items():
            print_tree(child_node, goal_values, depth=depth+1, requirement=req_path)

# Implementasi fungsi perhitungan entropi dengan formula ∑ -p log2(p)
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


def information_gain(data, target_attribute, attribute, split=False):
    size = len(data) # |S|
    entropy_s = entropy(data, target_attribute) # Entropy(S)
    entropy_single = 0
    split_entropy = 0

    # Handle discrete value
    if isinstance(attribute, str):
        single_attribute = data[attribute].unique()
        for attr in single_attribute:
            s_attr = data[data[attribute] == attr]

            # Alternative measures for selecting attributes: gain ratio.
            if split:
                # Calculate split entropy = - ∑ (|Sv| / |S|) log2 (|Sv| / |S|) 
                split_entropy += (-1.0) * ((len(s_attr) * 1.0 / size) * log2(len(s_attr) * 1.0 / size))

            entropy_temp = entropy(s_attr, target_attribute)
            entropy_single += len(s_attr) * entropy_temp / size


    # Handle continuous value
    else: # attribute = (sepal, 2.5)
        attr_name, split_point = attribute

        s_attr = data[data[attr_name] <= split_point]
        entropy_temp = entropy(s_attr, target_attribute)
        
        if size != 0:
            entropy_single += len(s_attr) * entropy_temp / size

        s_attr = data[data[attr_name] > split_point]
        entropy_temp = entropy(s_attr, target_attribute)
        
        if size != 0:
            entropy_single += len(s_attr) * entropy_temp / size

    gain = entropy_s - entropy_single

    # Alternative measures for selecting attributes: gain ratio.
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

'''
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A <- The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, v_i, of A,
            Add a new tree branch below Root, corresponding to the test A = v_i.
            Let Examples(v_i) be the subset of examples that have the value v_i for A
            If Examples(v_i) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(v_i), Target_Attribute, Attributes – {A})
    End
    Return Root
'''
def id3_build_tree(examples, goal_values, target_attribute, attributes):
    root = Node()
    children_label = {}

    # Program dapat menghandle banyak target. Bukan hanya positive dan negative
    for gv in goal_values:
        if (examples[target_attribute] == gv).all():
            root.label = gv
            children_label[gv] = len(examples)

            for i in range(len(goal_values)):
                if (goal_values[i] != gv):
                    children_label[goal_values[i]] = 0

            root.children_label = children_label
            
            return root

    for i in range(len(goal_values)):
        children_label[goal_values[i]] = 0

    if len(attributes) == 0:
        mode = examples[target_attribute].mode()
        if len(mode) > 0:
            root.label = examples[target_attribute].mode()[0]
        
        root.children_label = children_label
        return root
        
    ig = []
    for attribute in attributes:
        ig.append(information_gain(examples, target_attribute, attribute))
    
    A = attributes[ig.index(max(ig))]
    root.value = A

    if isinstance(A, str): # DISCRETE
        values = examples[A].unique()
        for vi in values:
            examples_vi = examples[examples[A] == vi]
            
            if examples.empty:
                new_node = Node()
                new_node.label = examples[target_attribute].mode()[0]
            
            else:
                next_attr = []
                for attr in attributes:
                    if attr != A: 
                        next_attr.append(attr)

                new_node = id3_build_tree(examples_vi, goal_values, target_attribute, next_attr)

                for i in range(len(goal_values)):
                    children_label[goal_values[i]] += new_node.children_label[goal_values[i]]
 
            root.children.update({vi: new_node})

        root.children_label = children_label
        return root
    
    else: #CONTINUOUS
        attr_name, split_point = A

        # Lebih kecl (<=)
        examples_vi = examples[examples[attr_name] <= split_point]
        
        if examples.empty:
            new_node = Node()
            new_node.label = examples[target_attribute].mode()

        else:
            new_node = id3_build_tree(
                examples_vi, 
                goal_values, 
                target_attribute, 
                [i for i in attributes if (i != A and i[1] <= split_point)]
            )
            root.children.update({'<=' + str(split_point): new_node})

            for i in range(len(goal_values)):
                children_label[goal_values[i]] += new_node.children_label[goal_values[i]]

        # Lebih besar (>)
        examples_vi = examples[examples[attr_name] > split_point]
        
        if examples.empty:
            new_node = Node()
            new_node.label = examples[target_attribute].mode()

        else:
            new_node = id3_build_tree(
                examples_vi, 
                goal_values, 
                target_attribute, 
                [i for i in attributes if (i != A and i[1] > split_point)]
            )
            root.children.update({'>' + str(split_point): new_node})

            for i in range(len(goal_values)):
                children_label[goal_values[i]] += new_node.children_label[goal_values[i]]

        root.children_label = children_label
        return root

# Mengembalikan target dari decision tree yang telah dibuat jika menerima masukan sebuah data uji
def id3_classify(root, example):
    while root.label == 'Node':
        if (isinstance(root.value, str)):
            if example[root.value] in root.children:
                root = root.children[example[root.value]]
            else:
                return
        
        else:
            val, split_point = root.value
            if (example[val] <= split_point):
                root = root.children['<=' + str(split_point)]

            else:
                root = root.children['>' + str(split_point)]
            
    return root.label

# Mengembalikan tingkat keakuratan dari decision tree jika menerima beberapa data uji
def id3_correctness(root, data_test, target_attribute):
    size = len(data_test)
    correct = 0
    
    for i, _ in data_test.iterrows():
        tes_result = data_test.loc[i, target_attribute]
        id3_result = id3_classify(root, data_test.loc[i, :])
        
        if tes_result == id3_result:
            correct += 1

    return correct / size

# Pruning dilakukan dengan mengubah daun dari sebuah node menjadi mayoritas daun-daun anak-anaknya
# Selanjutnya dilakukan pengecekan tingkat keakuratan antara decision tree yang awal dengan setelah prunning
# Jika keakuratan decision tree setelah dilakukan prunning lebih tinggi, maka pruning akan dilakukan
def id3_prune(examples, target_attribute, root, tree):
    if root.label != 'Node':
        return root
    
    for val in root.children.keys():
        root.children[val] = id3_prune(examples, target_attribute, root.children[val], tree)
    
    old_correctness = id3_correctness(tree, examples, target_attribute)
    
    temp_max = 0
    most_target = ''
    
    for label in root.children_label:
        if (root.children_label[label] > temp_max):
            temp_max = root.children_label[label]
            most_target = label
        
    root.label = most_target

    new_correctness = id3_correctness(tree, examples, target_attribute)

    if new_correctness > old_correctness:
        return root
    
    else:
        root.label = 'Node'
        return root
    
def split_data_set(data, fraction):
    train_set = data.sample(frac=0.75, random_state=0)
    test_set = data.drop(train_set.index)

    return train_set, test_set

def get_attributes_and_split(df, attr_name):
    attributes_and_split = []
    values = sorted(list(set(df[attr_name])))
    
    for i in range(len(values) - 1):
        split_point = round((values[i] + values[i+1]) / 2, 2)
        attributes_and_split.append((attr_name, split_point))
    
    return attributes_and_split

def replace_missing_atribute(df):
    row = df.shape[0]
    cols = list(df.columns)
    
    for i in range(0, row):
        for col in cols:
            if (df.at[i, col] == '?'):
                target = df.at[i, cols[-1]]
                temp_df = df[df[cols[-1]] == target]
                most_common_attr_values = temp_df.mode()
                # nilai yang diassign
                col_values = most_common_attr_values[col][0]
                df.at[i, col] = col_values
    
    return df

def ID3(df):
    df = replace_missing_atribute(df)
    target_attribute = list(df.columns)[-1]
    attributes_name = list(df.columns)
    attributes_name.pop(-1)

    goal_values = df[target_attribute].unique()

    attributes = []
    for attr in attributes_name:
        if isinstance(df[attr][0], str):
            attributes += attributes_name
            break
        else:
            attributes.extend(get_attributes_and_split(df, attr))

    data_example, data_pruning = split_data_set(df, 0.2)

    id3_tree = id3_build_tree(data_example, goal_values, target_attribute, attributes)
    id3_tree = id3_prune(data_pruning, target_attribute, id3_tree, id3_tree)
    print_tree(id3_tree, goal_values)

    # data_tes = pd.read_csv("iris-tes.csv")
    # print("Tingkat keakuratan setelah pruning:", id3_correctness(id3_tree, data_tes, target_attribute))
