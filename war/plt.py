#coding=utf-8
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
myfont = fm.FontProperties(fname='C:/Windows/Fonts/Deng.ttf')
import numpy as np
import pandas as pd
df = pd.read_csv("E:\pyBusiness\war\data.csv", header=None)
y = df.loc[0:100, 4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o',label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x',label='versicolor')
plt.xlabel('花瓣的长度',fontproperties=myfont)
plt.ylabel('花径的长度',fontproperties=myfont)
plt.legend(loc = 'upper left')
plt.show()
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    marker = ('s','x','o','v')
    colors = ('red','blue', 'Lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() -1,X[:,0].max()
    x2_min, x2_max = X[:, 1].min() -1,X[:,1].max()
    print(x1_min, x1_max)
    print(x2_min, x2_max)
plot_decision_regions(X, y, ppn, resolution = 0.02)
