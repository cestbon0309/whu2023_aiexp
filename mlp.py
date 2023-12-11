import os
import struct
import math
import numpy as np
import pickle


def load_mnist_data(kind):
    '''
    加载数据集
    :param kind: 加载训练数据还是测试数据
    :return: 打平之后的数据和one hot编码的标签
    '''
    labels_path = './data/%s-labels.idx1-ubyte' % kind
    images_path = './data/%s-images.idx3-ubyte' % kind
    with open(labels_path, 'rb') as lbpath:
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)
    with open(images_path, 'rb') as imgpath:
        struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images / 255., np.eye(10)[labels]

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def leaky_relu(z):
    '''
    leaky relu激活函数
    :param z: 神经网络的输出
    :return: z激活之后的值
    '''
    return np.where(z > 0, z, z * 0.01)


def leaky_relu_prime(z):
    '''
    leaky relu激活函数的导数
    :param z: 神经网络的输出
    :return: 关于z的导数
    '''
    z_ = np.copy(z)
    z_[z > 0] = 1
    z_[z < 0] = 0.01
    z_[z == 0] = 0.5
    return z_


def calc_loss(result,tag):
    ret = 0
    for x in range(10):
        ret += result[x]*result[x] if x!=int(tag) else (1-result[x])*(1-result[x])

def cross_entropy_loss(y_predict, y_true):
    """
    交叉熵损失函数
    :param y_predict: 预测值,shape (N,d)，N为批量样本数
    :param y_true: 真实值,shape(N,d)
    :return:
    """
    y_exp = np.exp(y_predict)
    y_probability = y_exp / np.sum(y_exp, axis=-1, keepdims=True)
    loss = np.mean(np.sum(-y_true * np.log(y_probability), axis=-1))  # 损失函数
    dy = y_probability - y_true
    return loss, dy


def plot_trainning(order1, order2, img_name):
    '''
    画出训练过程的对比图
    :param order1: 第一种网络结构
    :param order2: 第二种网络结构
    :param img_name: 图片名称
    :return:
    '''
    with open(order1, 'rb') as f1, open(order2, 'rb') as f2:
        accs1 = pickle.load(f1)
        accs2 = pickle.load(f2)

    import matplotlib.pyplot as plt
    plt.figure()
    # x = [str(i) for i in range(1, len(accs1) + 1)]
    x = [i for i in range(1, len(accs1) + 1)]
    plt.plot(x, accs1, label=order1)
    plt.plot(x, accs2, label=order2)
    plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)


def plot_single_training(order,mlp_obj, img_name='best_acc.png'):
    '''
    画出最优参数下的训练过程
    :param order:
    :param img_name:
    :return:
    '''
    with open(order, 'rb') as f1:
        accs = pickle.load(f1)
    import matplotlib.pyplot as plt
    plt.figure()
    x = [i for i in range(1, len(accs) + 1)]
    plt.plot(x, accs)
    #plt.plot(x, mlp_obj.training_loss)
    # plt.legend()
    # plt.ylim((0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(img_name)


class MLP_NeuralNet:
    def __init__(self,size=[784,1024,64,10]) -> None:
        self.sizes = size
        self.num_layers = len(size)
        weights_scale = 0.01
        self.weights = [np.random.randn(ch1, ch2) * weights_scale for ch1, ch2 in zip(size[:-1], size[1:])]
        self.biases = [np.random.randn(1, ch) * weights_scale for ch in size[1:]]
        self.training_loss = []

    def forward(self,x):
        self.X = [x]
        self.Z = []
        for layer_idx,(b,w) in enumerate(zip(self.biases,self.weights)):
            z = np.dot(x,w) + b
            x = leaky_relu(z)
            self.X.append(x)
            self.Z.append(z)
        return self.X[-1]

    def backward(self,y):#y = ground truth
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        loss,delta = cross_entropy_loss(self.Z[-1],y)
        self.training_loss.append(loss)
        batch_size = len(y)
        for l in range(self.num_layers - 2, -1, -1):
            x = self.X[l]

            db[l] = np.sum(delta, axis=0) / (batch_size)
            dw[l] = np.dot(x.T, delta) / batch_size

            if l > 0:
                # delta = np.dot(delta, self.weights[l].T) * sigmoid_prime(self.Z[l - 1])
                # delta = np.dot(delta, self.weights[l].T) * relu_prime(self.Z[l - 1])
                delta = np.dot(delta, self.weights[l].T) * leaky_relu_prime(self.Z[l - 1])
        return dw, db
        
    def update_para(self, dw, db, lr, l1=0, l2=0):
        if l1 != 0:
            # L1范数正则化
            self.weights = [w - lr * (nabla + l1 * np.sign(w)) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        elif l2 != 0:
            # L2范数正则化
            self.weights = [w - lr * (nabla + l2 * w) for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]
        else:
            # 不进行正则化
            self.weights = [w - lr * nabla for w, nabla in zip(self.weights, dw)]
            self.biases = [b - lr * nabla for b, nabla in zip(self.biases, db)]

    def evaluate(self,test_imgs,test_labels):
        result = []
        n = len(test_imgs)
        for batch_indx in range(0, n, 128):
            lower_range = batch_indx
            upper_range = batch_indx + 128
            if upper_range > n:
                upper_range = n
            test_x = test_imgs[lower_range: upper_range, :]
            result.extend(np.argmax(self.forward(test_x), axis=1))
        tlabel = np.argmax(test_labels,axis=1)
        correct = sum(int(pred == y) for pred, y in zip(result, tlabel))
        return correct

    
    def train(self,train_imgs,train_labels,test_images,test_labels,epochs=1000,lr = 0.1,l2=0,batch_size = 128,l1 = 0,orders='first', gamma=1, step_size=0):
        lr0 = lr
        n_train = len(train_imgs)
        n_test = len(test_images)
        accs = []
        for epoch in range(epochs):
            for batch_index in range(0,n_train,batch_size):
                lower = batch_index
                upper = batch_index+batch_size
                upper = n_train if upper > n_train else upper
                train_i = train_imgs[lower:upper,:]
                train_lb = train_labels[lower:upper]
                self.forward(train_i)
                dw,db = self.backward(train_lb)
                self.update_para(dw,db,lr,l1=l1,l2=l2)
            print(lr,end = '\t')
            if step_size != 0:
            # 阶梯式衰减
                if (epoch + 1) % step_size == 0:
                    lr *= gamma
            elif gamma != 1:
            # 指数衰减
                lr = math.pow(gamma, epoch) * lr0
            acc = self.evaluate(test_images, test_labels)
            accs.append(acc / 10000.0)
            print('Epoch {0}: {1} / {2}'.format(epoch, acc / 10000.0, n_test))
            with open(orders, 'wb') as f:
                pickle.dump(accs, f)
        plot_single_training(orders,self)
    
    def save_param(self):
        with open('weight','wb') as f:
            pickle.dump(self.weights,f)
        with open('bias','wb') as f:
            pickle.dump(self.biases,f)
    
    def load_param(self):
        with open('weight','rb') as f:
            self.weights = pickle.load(f)
        with open('bias','rb') as f:
            self.biases = pickle.load(f)


if __name__ == '__main__':
    mlp = MLP_NeuralNet()
    train_imgs,train_labels = load_mnist_data('train')
    test_imgs,test_labels = load_mnist_data('t10k')
    mlp.train(train_imgs,train_labels,test_imgs,test_labels,epochs=20)
    mlp.save_param()
    #mlp.load_param()
    #correct = mlp.evaluate(test_imgs,test_labels)
    #print(f'acc {correct/10000.0}')