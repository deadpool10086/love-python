import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
class AdalineGD(object):
    """
    eta:flaot
    学习效率，处于0和1

    n_iter:int
    对训练数据进行学习改进次数

    w_:一维向量
    存储权重数值

    error_:
    存储器每次迭代是，网络对数据进行错误判断的次数
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_iter = n_iter
    def fit(self, X, y):
        """
        X:二维数组[n_sampls, n_features]
        n_samples 表示X中含有训练数据条目数
        n_feature 含有4个数据的以为向量，用于表示一条训练条目

        y:一维向量
        用于存储每一训练条目对应的正确分类

        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            output = self.net_input(X)
            #output = w0 + w1*x1 + w2*x2 .... wn*xn
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            print(errors)
            input();
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self
    def net_input(self, X):
        return np.dot(X,self.w_[1:]) + self.w_[0]
    def activation(self, X):
        return self.net_input(X)
    def predict(self, X):
        return np.where(self.activation(X) >= 0, 1, -1)


df = pd.read_csv("D:\pywork\war\data.csv", header=None)
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values
plt.scatter(X[:50,0], X[:50,1], color='red', marker='o',label='setosa')
plt.scatter(X[50:100,0], X[50:100,1], color='blue', marker='x',label='versicolor')
plt.xlabel('花瓣的长度')
plt.ylabel('花径的长度')
plt.legend(loc = 'upper left')
plt.show()

from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution = 0.02):
    markers = ('s','x','o','v')
    colors = ('red','blue', 'Lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() -1,X[:,0].max()
    x2_min, x2_max = X[:, 1].min() -1,X[:,1].max()
    xx1, xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),np.arange(x2_min,x2_max,resolution))
    z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    plt.contour(xx1,xx2,z,alpha=0.4,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0], y=X[y==cl,1], alpha=0.8, c=cmap(idx),marker=markers[idx], label = cl)



ada = AdalineGD(eta = 0.0001, n_iter=50)
ada.fit(X,y)
plot_decision_regions(X,y,classifier = ada)
plt.title('Adaline-Gradient descent')
plt.xlabel('花茎长度')
plt.ylabel('花瓣长度')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('sum-squard-error')
plt.show()
