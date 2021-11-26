import pandas as pd
import math
from matplotlib import pyplot as plt 
import numpy as np
import copy

# 定义网络
class Network(object):
    def __init__(self, num_of_weights):
        # 随机产生w的初始值
        # 为了保持程序每次运行结果的一致性，
        # 此处设置固定的随机数种子
        np.random.seed(0)
        self.w = np.random.randn(num_of_weights, 1) #模拟我们上述的随机产生初始化的w
        self.b = 0.
    
    # 这就是我们的前向函数(计算一次x*w+b)
    def forward(self, x):
        y_pred = np.dot(x, self.w) + self.b
        return y_pred
    
    # 计算预测的y和真实的y值的损失函数
    def loss(self, y, y_pred):
        cost = (y-y_pred)*(y-y_pred)
        cost = np.mean(cost)
        return cost
    
    # 计算梯度
    def gradient(self, x, y):
        y_pred = self.forward(x)
        # 计算w的梯度
        gradient_w_sum = (y_pred - y) * x
        gradient_w = np.mean(gradient_w_sum, axis = 0)
        gradient_w = gradient_w.reshape(self.w.shape) # 梯度维度跟w要保持一致
        
        # 计算b的梯度
        gradient_b_sum = (y_pred - y)
        gradient_b = np.mean(gradient_b_sum) #这里不指定行和列，计算出一个总平均的实数，跟我们定义的b是一致的，都是实数
        return gradient_w, gradient_b
    
    # 更新模型参数, w和b的值
    def update(self, gradient_w, gradient_b, learning_rate = 0.01):
        self.w = self.w - learning_rate * gradient_w
        self.b = self.b - learning_rate * gradient_b
        
# 数据归一化操作
def normalize(data):
    result_data = copy.deepcopy(data)
    # data[:,i]代表取第i列的数据
    for i in range(0,data.shape[1]):
        result_data[:,i] = ((data[:,i] - np.mean(data[:,i]))/np.std(data[:, i]))
    return result_data

# 导入数据，并做数据预处理
def load_data_and_preprocess():
    # 读取数据
    data = pd.read_csv('./boston_housing_data.csv')
    columns = list(data.columns)

    # 获取除去标签列的特征数据集
    feature_data = data.drop(['MDEV'], axis=1)
    # 获取标签数据集
    target_data = data['MDEV']

    # 将标签的df数据转换成numpy的数组类型
    feature_data = np.array(feature_data, dtype=float)
    # 特征数据预处理-归一化操作
    normalize_feature_data =  normalize(feature_data)
    
    # 将特征的df数据转换成numpy的数组类型
    target_data = np.array(target_data, dtype=float)
    target_data = target_data.reshape(len(target_data), 1)
    
    # 为了更便于后续的表示和讲解，这里我们把特征数据统一用x表示
    x = normalize_feature_data
    # 为了更便于后续的表示和讲解，这里我们把标签数据统一用y表示
    y = target_data
    return x, y

# 按照一定比例，切分数据集为训练数据集和测试数据集
# 训练集用于模型训练
# 测试集用于模型测试
def split_data(x, y, radio=0.8):
    train_sample_num = math.ceil(len(x)*0.8)
    
    x_train = x[:train_sample_num]
    x_test = x[train_sample_num:]
    
    y_train = y[:train_sample_num]
    y_test = y[train_sample_num:]
    return x_train, x_test, y_train, y_test
    

if __name__=="__main__":
    # 导入数据并做数据预处理操作
    x, y = load_data_and_preprocess()
    
    # 按照特定比例进行训练集和测试集的切分
    x_train, x_test, y_train, y_test = split_data(x, y, 0.8)
    
    # 定义网络
    net = Network(13) #特征数为13
    learning_rate = 0.01 # 定义学习率
    epochs = 5000 # 定义训练轮数
    
    losses = []
    # 迭代训练过程
    for i in range(epochs):
        # 执行前向函数，计算模型预测值
        y_pred = net.forward(x_train)
        # 通过损失函数，计算loss值
        loss = net.loss(y_train, y_pred)
        losses.append(loss)
        
        if i % 100 == 0:
            print(f'epochs:{i} | loss:{loss}')
        # 计算梯度
        gradient_w, gradient_b = net.gradient(x, y)
        # 根据梯度下降法，更新模型参数
        net.update(gradient_w, gradient_b)
    
    # 绘制loss曲线
    # 画出损失函数的变化趋势
    plot_x = np.arange(epochs)
    plot_y = np.array(losses)
    plt.plot(plot_x, plot_y)
    plt.show()
    
    # 遍历测试集每个样本(即每行)
    for i in range(len(x_test)):
        one_test_sample = x_test[i]
        one_test_lable = y_test[i]

        # 模型前向函数
        y_test_pred = net.forward(one_test_sample)
        print(f'[real house price/predicate house price]:[{y_test[i][0]} / {y_test_pred[0]}]')

    