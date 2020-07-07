import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('input/train.csv')
Y = df['species'].values
X = df.drop(['species'], axis=1).values

Y_class = np.unique(Y).tolist() # 所有分类，索引为其量化值
Y = np.asarray([Y_class.index(y) for y in Y])

theta = np.zeros(X.shape[0])

# 核函数(线性)
def kernel(t, x):
    m = np.empty(x.shape).T
    
    return x

print()