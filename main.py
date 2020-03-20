import pandas as pd
from id3 import *

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


df = pd.read_csv("iris-full.csv")

# splitting dataset
df_train = df.sample(frac=0.9, random_state=200)
df_test = df.drop(df_train.index)

id3_result = ID3(df_train)  # why error if use df_train

# predict
pred_y = predict(id3_result, df_test)
real_y = df_test[df_test.columns[-1]].values.tolist()

results = confusion_matrix(pred_y, real_y)

print('Confusion Matrix :')
print(results)
print('Accuracy Score :', accuracy_score(pred_y, real_y) * 100, "%")
print('Report : ')
print(classification_report(pred_y, real_y))
