from pprint import pprint
from sklearn import tree
from sklearn.datasets import load_iris

# load dataset
iris = load_iris()
pprint(iris)

# building model
clf_iris = tree.DecisionTreeClassifier()
clf_iris.fit(iris.data, iris.target)

# presenting model
r = tree.export_text(clf_iris, iris.feature_names)
print(r)
