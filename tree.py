from pprint import pprint


class Node:
    def __init__(self, value=None, children=None):
        # value = this node data/value
        # children = {'path_name_1': Node(), 'path_name_2': Node(), ...}
        self.value = value
        self.children = children

    def add_child(self, path_name, child_node):
        # path_name = what is requirement to go to child_node
        # child_node = another tree
        self.children[path_name] = child_node

    def print_tree(self, depth=0, indent=4, requirement=None):
        if requirement is None:
            print("{}{}".format(" "*(indent*depth), self.value))
        else:
            if self.children is None:
                str_format = "{} == {} --> {}"
            else:
                str_format = "{} == {} --> ({} ?)"
            print(str_format.format(" " * (indent*depth), requirement, self.value))
        if self.children is not None:
            for req_path, child_node in self.children.items():
                child_node.print_tree(depth=depth+1, requirement=req_path)


# example how to use
if __name__ == "__main__":
    root = Node('outlook', {
        'sunny': Node('humidity', {
            'high': Node('no'),
            'normal': Node('yes')
        }),
        'overcast': Node('yes'),
        'rain': Node('wind', {
            'strong': Node('no'),
            'weak': Node('yes')
        })
    })

    root.print_tree()
