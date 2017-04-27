import numpy as np
class Perceptron(object):
    """
    eta : 学习率
    n_iter:权重向量训练次数
    w_: 神经分叉权重向量
    errors:用于记录神经元判断出错的次数

    """

    def __init__(self,eta = 0.1, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter
        pass
    def fit(self,X,y):
        """
        输入训练数据,培训神经元，x输入样本向量，y对应样本分类
        X:shape[n_samples, n_features]
        x:[[1,2,3],[4,5,6]]
        n_samples: 2
        n_features: 3
        y:[1,-1]
        """

        """
        初始换权重向量为0
        加一是因为前面算法提到的w0,也就是阈值
        """
        self.w_ = np.zero(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            """
            X:[[1,2,3],[4,5,6]]
            y:[1,-1]
            zip[X,y] = [[1,2,3, 1],[4,5,6, -1]]
            """
            for xi, target in zip(X,y):
                """
                update = η * (y -y')
                """
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
                self.errors_.append(errors)
    def net_input(self, x):
        """
        z = w0*l w1*w1 +
        """
        return np.self.w_[1:] + self.w_[0]

    def predict(self, x):
        return np.where(self.net_input(x) >= 0.0 , 1, -1)
