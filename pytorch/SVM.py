import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.optim import SGD
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# %matplotlib inline

X_train = np.array([[3,3],
                    [4,3],
                    [1,1]])
y_train = np.array([1,1,-1])
X_test = X_train
y_test = y_train

# 定义感知机/SVM 网络层
class Perceptron(nn.Module):
    def __init__(self):
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(2, 1)
        
    def forward(self, x):
        x = self.layer(x)
        return x

# 计算感知机损失
def loss_func(scores, label, type="svm"):
    assert type=="perceptron" or type=="svm", "loss type error"
    if type == "perceptron":
        # 感知机损失函数，label取值集合为{-1, 1}
        loss = -label*scores
    else:
        # SVM损失函数，label取值集合为{-1, 1}
        loss = 1-label*scores
   
    loss[loss<=0] = 0
    return torch.sum(loss)

# 定义激活函数，注意：计算损失时，不经过激活层
def sign(x):
    x[x>=0] = 1
    x[x<0] = -1
    return x

# 预测函数
def pred(x):
    return sign(x)

model = Perceptron()

optim_func = SGD(model.parameters(), lr=0.01)

# 训练开始前模型初始参数
for name,param in model.named_parameters():
    print(name, param)

for epoch in range(10000):
    inputs, targets = X_train, y_train
    inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False)
    label = Variable(torch.from_numpy(targets).int(), requires_grad=False)
    # 前向传播
    scores = model(inputs).squeeze(1)
    loss = loss_func(scores, label, "perceptron")
    # 反向传播
    optim_func.zero_grad()
    loss.backward()
    optim_func.step()
    if epoch % 1000 == 0:
        # 计算分类的准确率
        inputs, targets = X_test, y_test
        inputs = Variable(torch.from_numpy(inputs).float(), requires_grad=False)
        label = Variable(torch.from_numpy(targets).int(), requires_grad=False)
        scores = model(inputs).squeeze(1)
        num_correct = (pred(scores) == label).sum().item()
        acc = num_correct*100.0 / inputs.shape[0]
        print("loss=",loss.detach().numpy(),"acc=", acc)
        for name,param in model.named_parameters():
            print(name, param)

# 训练结束后模型参数
for name,param in model.named_parameters():
    print(name, param)

def plot_predict(x, w, b):
        return np.dot(w, x) + b

def plot_decsion_plane(X_data, y_data, w, b):
    # 画决策面
    colors = ['red', 'blue']
    cmap = ListedColormap(colors[:len(np.unique(y_data))])
    x1_min, x1_max = X_data[:, 0].min() - 1, X_data[:, 0].max() + 1
    x2_min, x2_max = X_data[:, 1].min() - 1, X_data[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02),np.arange(x2_min, x2_max, 0.02))
    Z = plot_predict(np.array([xx1.ravel(), xx2.ravel()]), w, b)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # 画样本点
    markers = ('x', 'o')
    for idx, ci in enumerate(np.unique(y_data)):
        plt.scatter(x=X_data[y_data == ci, 0], y=X_data[y_data == ci, 1], alpha=0.8, c=np.atleast_2d(cmap(idx)), 
                    marker=markers[idx], label=ci)

    # 画图例
    plt.legend(loc='upper left')
    plt.show()

plot_decsion_plane(X_test,y_test,model.state_dict()['layer.weight'].numpy(),
                  model.state_dict()['layer.bias'].numpy())
    