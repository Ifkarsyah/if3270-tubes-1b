import pandas as pd
from id3 import *

df = pd.read_csv("your_data_here.csv")
# df = df.drop("day", 1)
ID3(df)
