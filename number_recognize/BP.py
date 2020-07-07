import numpy as np
import matplotlib.pyplot as plt
import os
import json
import datetime
import math

# 生成神经网络结构
def generate_l(feature_num, K, hidden_num, hidden_cell_num):
    l = list(range(hidden_num + 2))
    for i, v in enumerate(l):
        if(i == 0): l[i] = feature_num
        elif(i == hidden_num + 1): l[i] = K
        else: l[i] = hidden_cell_num
    return l

# 生成权重矩阵
def generate_theta(l):
    L = len(l)
    theta_l = list(range(L - 1))
    for i in range(L - 1):
        t_1 = l[i] + 1
        t_2 = l[i + 1]
        epsilon = (6.0/(t_1 + t_2)) ** 0.5
        t = np.random.rand(t_1, t_2)*2*epsilon - epsilon
        theta_l[i] = t
    return theta_l


def sigmoid(theta, x):
    # 权重函数
    _x = np.c_[np.ones((x.shape[0], 1)), x] # 添加偏置单元
    z = np.dot(theta.T, _x.T).T
    # 逻辑回归
    return 1 / (1 + np.exp(-z))

# 代价函数
def generate_J(g_l, Y, labd, theta_l):
    m = Y.shape[0]
    # K
    _h = g_l[-1]
    e_1 = np.multiply(Y, np.log10(_h))
    e_2 = np.multiply((1 - Y), np.log10(1 - _h))
    e = (-1/m) * np.sum(e_1 + e_2)
    reg = 0 # 正则项
    if(labd > 0):
        reg = sum([np.sum(t[1:] ** 2) for t in theta_l])
    return e + labd/(2*m) * reg


# 残差函数
def generate_D_l(theta_l, g_l, Y, labd):
    L = len(theta_l) + 1
    M = Y.shape[0]
    d_l = list(range(L)) # delta 0层无误差
    d_l[-1] = g_l[-1] - Y
    d_l[-1] = np.c_[np.zeros((d_l[-1].shape[0], 1)), d_l[-1]]
    for i in range(2, L):
        j = L - i
        _a = np.multiply(g_l[j], 1 - g_l[j])
        d_l[j] = np.multiply(np.dot(d_l[j + 1][:,1:], theta_l[j].T), _a)

    D_l = [np.zeros(t.shape) for t in theta_l]
    for i in range(len(D_l)):
        for m in range(M):
            _d = np.dot(g_l[i][m].reshape(-1,1), d_l[i + 1][:,1:][m].reshape(1,-1))
            D_l[i] = D_l[i] +_d
        # 正则化 偏置节点不参与正则化
        D_l[i][0] = D_l[i][0] / M
        D_l[i][1:] = D_l[i][1:] / M + labd * theta_l[i][1:]
    return D_l

# 拆分样本为指定batch
def get_batches(sample, size = 50):
    b_len = math.ceil(len(sample)/size)
    return [sample[b*size:(b+1)*size] for b in range(b_len)]

# 洗牌
def shuffle(x, y):
    y_n = y.shape[1]
    x_y = np.c_[x, y]
    np.random.shuffle(x_y)
    return x_y[:,:-y_n], x_y[:,-y_n:]

class BP(object):
    """
    Parameters
    -----------
    X: 样本集（m x feature_num）
    Y: 结果集（m x K）
    hidden_num: 隐藏层数
    hidden_cell_num: 隐藏层节点数
    iter_num: 迭代次数
    labd: 正则化系数
    gama: 加速梯度动量系数
    alpha: 梯度下降学习率
    threshold: 迭代优化率阈值
    is_save_module: 是否保存该次训练的模型
    is_use_module: 是否使用当前目录下的模型
    module_name: 保存的模型文件夹名

    Propertys
    -----------
    fit: 训练器

    """
    def __init__(self, X_train, Y_train, X_valid = None, Y_valid = None, X_test = None, Y_test = None, K = 1, hidden_num = 1, hidden_cell_num = 5, iter_num = 100, alpha = 0.01, labd = 0.0001, gama = 0.9, threshold = 0, is_save_module = False, is_use_module=False, module_name = 'module_BP'):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_valid = X_valid
        self.Y_valid = Y_valid
        self.X_test = X_test
        self.Y_test = Y_test
        self.K = K
        self.hidden_num = hidden_num
        self.hidden_cell_num = hidden_cell_num
        self.iter_num = iter_num
        self.labd = labd
        self.alpha = alpha
        self.gama = gama
        self.threshold = threshold
        self.is_save_module = is_save_module
        self.is_use_module = is_use_module
        self.module_name = module_name
        self.acc_dict = {} # 记录预测的正确率
        self.l = generate_l(X_train.shape[1], K, hidden_num, hidden_cell_num)
        # 是否使用已有模型
        if(is_use_module): 
            if(os.path.exists(module_name)): self.theta_l = self.use_module()
            else:
                print('无可用模型，已随机初始化')
                self.theta_l = generate_theta(self.l)
        else: self.theta_l = generate_theta(self.l)
        # 加速梯度下降剩余动量
        self.momentum = [np.zeros(t.shape) for t in self.theta_l]
    
    def g(self, x, t = None):
        if(t is None): t = self.theta_l
        L = len(t) + 1
        g_l = list(range(L))
        g_l[0] = x
        for i in range(L - 1):
            g_l[i + 1] = sigmoid(t[i], g_l[i])
            g_l[i] = np.c_[np.ones((g_l[i].shape[0], 1)), g_l[i]] # 偏置单元
        return g_l
    
    def J(self, x, y, other_theta_l = None):
        theta_l = self.theta_l
        g_l = self.g(x)
        if(not (other_theta_l is None)): theta_l = other_theta_l
        return generate_J(g_l, y, self.labd, theta_l)

    # 全量梯度下降
    def gd_normal(self):
        D_l = generate_D_l(self.theta_l, self.g(self.X_train), self.Y_train, labd=self.labd)
        for l, t in enumerate(self.theta_l):
            self.theta_l[l] = t - self.alpha * D_l[l]
    # 智能加速SGD
    def gd_NAG(self):
        # 洗牌
        _X, _Y = shuffle(self.X_train, self.Y_train)
        # 拆分样本为小批量
        X_batches = get_batches(_X, 50)
        Y_batches = get_batches(_Y, 50)
        for b, x in enumerate(X_batches):
            y = Y_batches[b]

            # 启动智能加速(NAG)
            _t_l = self.theta_l
            momentum_1 = list() # 预测下一次迭代的theta值
            for l, t in enumerate(self.momentum):
                momentum_1.append(_t_l[l] - self.gama * t)
            D_l_1 = generate_D_l(momentum_1, self.g(x, momentum_1), y, labd=self.labd)
            for l, t in enumerate(_t_l):
                _u = self.gama * self.momentum[l] + self.alpha * D_l_1[l]
                self.momentum[l]= _u # 更新动量
                t = t - _u

    # 训练器
    def fit(self, x = None, y = None):
        if((x is None) or (y is None)):
            x = self.X_train
            y = self.Y_train
        
        starttime = datetime.datetime.now() # 开始时间
        errors_train = []
        errors_valid = []

        def run(iter):
            if(iter <= 0): return
            # 验证集误差
            error_valie = self.J(self.X_valid, self.Y_valid)
            errors_valid.append(error_valie)
            # 训练集误差
            error_train = self.J(x, y)
            _p = 100
            if(len(errors_train) > 0): _p = (errors_train[-1] - error_train) / errors_train[-1] * 100
            print('第%i次优化：%.4f%%' % (self.iter_num - iter + 1, _p))
            errors_train.append(error_train)
            # 阈值
            if(error_train <= self.threshold): return

            # 梯度下降
            # self.gd_normal()
            self.gd_NAG()

            # 递归迭代
            run(iter - 1)
        run(self.iter_num)

        # 处理误差
        self.errors_train = errors_train
        self.errors_valid = errors_valid
        print('迭代次数：', len(errors_train))
        print('最终误差：', min(errors_train))

        # 是否保存当前模型
        if(self.is_save_module):
            self.save_module()

        # 预测
        self.predict()

        # 耗时
        print('训练耗时：%ss'%(datetime.datetime.now() - starttime).seconds)

        # 绘图
        plt.plot(errors_train, color='red', label='train')
        plt.plot(errors_valid, color='blue', label='valid')
        plt.legend(loc='upper right')
        plt.show()

        return errors_train
    
    # 预测
    def predict(self, x = None, y = None, text = 'acc'):
        if(x is None or y is None): 
            if(not (self.X_train is None) and not (self.Y_train is None)): self.predict(self.X_train, self.Y_train, 'acc_train')
            if(not (self.X_valid is None) and not (self.Y_valid is None)): self.predict(self.X_valid, self.Y_valid, 'acc_valid')
            if(not (self.X_test is None) and not (self.Y_test is None)): self.predict(self.X_test, self.Y_test, 'acc_test')
        else:
            P = self.g(x)[-1]
            f_p = lambda y: np.array([np.argmax(v) for v in y])
            P = f_p(P)
            y = f_p(y)
            M = len(y)
            acc = (P == y).sum() / M * 100
            print('%s：%.4f%%' % (text, acc))
            self.acc_dict[text] = acc # 记录预测的正确率
    
    # 梯度检验
    def check_gd(self, epsilon = 10 ** (-4)):
        x = self.X_train
        y = self.Y_train
        t_l = self.theta_l
        D_l = generate_D_l(self.theta_l, self.g(x), y, labd=self.labd)
        eva_gd_l = [np.zeros(x.shape) for x in t_l]
        for l, t in enumerate(self.theta_l):
            for i, t_1 in enumerate(t):
                for j, t_2 in enumerate(t_1):
                    t_l[l][i][j] = t_l[l][i][j] + epsilon
                    t_1 = self.J(x, y, t_l)
                    t_l[l][i][j] = t_l[l][i][j] - 2*epsilon
                    t_2 = self.J(x, y, t_l)
                    t_l[l][i][j] = t_l[l][i][j] + epsilon # 恢复
                    eva_gd_l[l][i][j] = (t_1 - t_2) / (2*epsilon)
        gd_p_l = list(range(len(eva_gd_l)))
        for l in range(len(eva_gd_l)):
            gd_p_l[l] = np.sum((eva_gd_l[l] - D_l[l]) ** 2) / eva_gd_l[l].size
        print('梯度检验的值为：', sum(gd_p_l))

    # 学习曲线
    def learn_curve(self):
        error_train = []
        error_valid = []
        for i in range(20):
            self.theta_l = generate_theta(self.l)
            error_train.append(self.fit(self.X_train[:(i+1)*10], self.Y_train[:(i+1)*10])[-1])
            self.theta_l = generate_theta(self.l)
            error_valid.append(self.fit(self.X_valid[:(i+1)*10], self.Y_valid[:(i+1)*10])[-1])
        plt.plot(range(10, 210, 10), error_train, color='red', label='train')
        plt.plot(range(10, 210, 10), error_valid, color='blue', label='valid')
        plt.legend(loc='upper right')
        plt.show()
    
    # 保存模型
    def save_module(self):
        if(not os.path.exists(self.module_name)): os.mkdir(self.module_name)
        json_name = '%s/info.json' % self.module_name
        
        # 判断是否可以优化模型
        if(os.path.exists(json_name)): 
            with open(json_name, 'r') as f:
                info = json.load(f)
                _pre = info['errors_train'] + info['errors_valid']
                _now = self.errors_train[-1] + self.errors_valid[-1]
                print('训练集+验证集误差（pre: %f，now：%f）'%(_pre, _now))
                if(_pre < _now): return # 无需优化模型

        with open(json_name, 'w') as f:
            theta_fs = list()
            for i, t in enumerate(self.theta_l):
                fn = '%s/theta_%i.csv'%(self.module_name, i)
                theta_fs.append({'index': i, 'filename': fn})
                np.savetxt(fn, t, delimiter=',') # 存为csv文件
            info = {
                'errors_train': self.errors_train[-1],
                'errors_valid': self.errors_valid[-1],
                'acc_dict': self.acc_dict,
                'theta_len': len(theta_fs),
                'theta_fs': theta_fs
            }
            json.dump(info, f)
            print('模型已保存')
    
    # 使用模型
    def use_module(self):
        json_name = '%s/info.json' % self.module_name
        with open(json_name, 'r') as f:
            info = json.load(f)
            theta_l = list(range(info['theta_len']))
            for t in info['theta_fs']:
                theta_l[t['index']] = np.loadtxt(t['filename'], delimiter=',')
            return theta_l
