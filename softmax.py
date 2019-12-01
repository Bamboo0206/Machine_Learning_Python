import random
import math
import numpy as np
import pandas as pd

class NeuralNetwork:
    LEARNING_RATE = 0.03 #学习率
    LAMBDA = 0.03 #正则化项参数

    # 初始化神经网络结构
    #单隐层神经网络，num_inputs, num_hidden, num_outputs分别是输入层、隐层、输出层节点数，后面几个参数是初始化权重和bias
    def __init__(self, num_inputs, num_hidden, num_outputs, hidden_layer_weights = None, hidden_layer_bias = None, output_layer_weights = None, output_layer_bias = None):
        self.num_inputs = num_inputs

        self.hidden_layer = NeuronLayer(num_hidden, hidden_layer_bias)
        self.output_layer = NeuronLayer(num_outputs, output_layer_bias)

        self.init_weights_from_inputs_to_hidden_layer_neurons(hidden_layer_weights)
        self.init_weights_from_hidden_layer_neurons_to_output_layer_neurons(output_layer_weights)

    #初始化输入层到隐层的权重
    def init_weights_from_inputs_to_hidden_layer_neurons(self, hidden_layer_weights):
        weight_num = 0
        for h in range(len(self.hidden_layer.neurons)):
            for i in range(self.num_inputs):
                if not hidden_layer_weights: #若未指定初始化参数，则随机初始化参数
                    self.hidden_layer.neurons[h].weights.append(random.random())
                else:
                    self.hidden_layer.neurons[h].weights.append(hidden_layer_weights[weight_num])
                weight_num += 1

    # 初始化隐层到输出层的权重
    def init_weights_from_hidden_layer_neurons_to_output_layer_neurons(self, output_layer_weights):
        weight_num = 0
        for o in range(len(self.output_layer.neurons)):
            for h in range(len(self.hidden_layer.neurons)):
                if not output_layer_weights: #若未指定初始化参数，则随机初始化参数
                    self.output_layer.neurons[o].weights.append(random.random())
                else:
                    self.output_layer.neurons[o].weights.append(output_layer_weights[weight_num])
                weight_num += 1

    # 打印调试信息，打印所有权重
    def inspect(self):
        print('------')
        print('* Inputs: {}'.format(self.num_inputs))
        print('------')
        print('Hidden Layer')
        self.hidden_layer.inspect()
        print('------')
        print('* Output Layer')
        self.output_layer.inspect()
        print('------')

    def feed_forward(self, inputs): #前馈
        self.hidden_layer_outputs = self.hidden_layer.feed_forward(inputs) #计算输入层到隐层
        self.output_layer_outputs = self.output_layer.softmax(self.hidden_layer_outputs)
        return self.output_layer_outputs

    # Uses online learning, ie updating the weights after each training case
    def train(self, training_inputs, training_outputs): #训练：前向传播计算输出和反向传播更新参数。参数：输入数据和输出数据
        self.feed_forward(training_inputs)

        # 3. Update output neuron weights
        self.hidden_layer_outputs = np.array(self.hidden_layer_outputs) #转ndarray
        for o in range(len(self.output_layer.neurons)):  # 每个输出节点 #每个输出节点对应一组权重 w_ho weight hidden to output
            pd_error_wrt_weight = (self.hidden_layer_outputs) * (self.output_layer_outputs[o] - training_outputs[o])#更新一组权值
            # self.output_layer.neurons[o].weights -= (self.LEARNING_RATE * pd_error_wrt_weight + self.LAMBDA * np.array(self.output_layer.neurons[o].weights)) #加了正则化项
            self.output_layer.neurons[o].weights -= self.LEARNING_RATE * pd_error_wrt_weight

        # # 4. Update hidden neuron weights
        a1=self.hidden_layer_outputs #应该是hidden layer的输出
        tmp=0
        for h in range(len(self.hidden_layer.neurons)):
            for o in range(len(self.output_layer.neurons)):
                tmp += (self.output_layer_outputs[o] - training_outputs[o]) * self.output_layer.neurons[o].weights[h] * (a1[h]*(1-a1[h]))#更新一组权值
            # self.hidden_layer.neurons[h].weights -= (self.LEARNING_RATE * tmp * training_inputs + self.LAMBDA * np.array(self.hidden_layer.neurons[h].weights)) #更新一组权值#加了正则化项
            self.hidden_layer.neurons[h].weights -= self.LEARNING_RATE * tmp * training_inputs

    def Cross_Entropy(self,x,y): # softmax损失函数：交叉熵函数 #x,y是整个数据集和标签。都是二维数组
        total_error = 0
        for i in range(x.shape[0]): #遍历所有样本 #这种写法对训练集效率稍微低一点点，算了两次forward
            self.feed_forward(x[i])
            for o in range(y.shape[1]):  # 对每一个输出单元计算error
                total_error -= y[i][o] * math.log(self.output_layer_outputs[o])
        return total_error

    def predict(self,x,y): #对一组数据预测输出，返回正确率
        correct = 0
        for i in range(x.shape[0]):  # 遍历所有样本 #这种写法对训练集效率稍微低一点点，算了两次forward
            self.feed_forward(x[i])
            self.get_onehot()
            if((self.get_onehot()==y[i]).all()):
                correct=correct+1
        print("correct=",correct)
        return correct/x.shape[0]

    def get_onehot(self): #将softmax输出转为onehot编码，用于计算正确率
        maxId = -1
        max_P = 0
        for i in range(self.output_layer_outputs.shape[0]):
            if(max_P<self.output_layer_outputs[i]):
                max_P = self.output_layer_outputs[i]
                maxId = i
        result=np.zeros(self.output_layer_outputs.shape[0])
        result[maxId]=1
        return result


class NeuronLayer: #神经网络的一层
    def __init__(self, num_neurons, bias): #初始化神经网络一层的神经元和权重

        # Every neuron in a layer shares the same bias
        self.bias = bias if bias else random.random()

        self.neurons = []
        for i in range(num_neurons): #为一层添加 num_neurons 个神经元
            self.neurons.append(Neuron(self.bias))

    def inspect(self): #打印调试信息
        print('Neurons:', len(self.neurons))
        for n in range(len(self.neurons)):
            print(' Neuron', n)
            for w in range(len(self.neurons[n].weights)):
                print('  Weight:', self.neurons[n].weights[w])
            print('  Bias:', self.bias)

    def feed_forward(self, inputs):  #前馈计算，计算一层神经元的输出
        outputs = []
        for neuron in self.neurons: #对每个神经元计算输出
            outputs.append(neuron.calculate_output(inputs))
        return outputs

    def softmax(self, inputs): #计算输出层softmax的输出
        outputs = [] #输出向量
        for neuron in self.neurons: #对每个神经元计算输出
            outputs.append(neuron.softmax_exp(inputs))
        output_sum=np.sum(outputs)
        # print("神经元输出：",outputs / output_sum)
        return outputs / output_sum


class Neuron: ##一个神经元，包含对应的一组权重
    def __init__(self, bias): #初始化权重和bias
        self.bias = bias
        self.weights = []

    def calculate_output(self, inputs): #计算一个神经元的输出（使用sigmoid函数
        self.inputs = inputs  #单个样本
        self.output = self.sigmoid(self.calculate_total_net_input())
        return self.output

    def softmax_exp(self, inputs): #计算softmax算式中的exp（ωx),还没有除以输出的和
        self.inputs = inputs  # 单个样本
        self.output = math.exp(self.calculate_total_net_input())
        return self.output

    def calculate_total_net_input(self): #加权求和,求z
        total = 0
        for i in range(len(self.inputs)): #加权求和
            total += self.inputs[i] * self.weights[i]
        return total + self.bias  # z

    # Apply the logistic function to sigmoid the output of the neuron
    def sigmoid(self, total_net_input): #sigmoid函数
        return 1 / (1 + math.exp(-total_net_input))


###
# 处理数据
def read_data():
    train = np.loadtxt('train.csv', delimiter=',')
    test = np.loadtxt('test.csv', delimiter=',')
    x_train = train[:, :2]
    y_train = train[:, 2]
    x_test = test[:, :2]
    y_test = test[:, 2]
    #return x_train,y_train,x_test,y_test
    return x_train, np.array(pd.get_dummies(y_train).values), x_test, np.array(pd.get_dummies(y_test).values) #标签y转onehot编码

from sklearn.model_selection import train_test_split #用于划分训练集和测试集
def read_data_thyroid(): #读取并处理数据集thyroid.txt
    data = np.loadtxt('thyroid.txt', delimiter=',')
    x_train,x_test,y_train,y_test = train_test_split(data[:,:21],data[:,21],test_size=0.3) #划分训练集和测试集
    return x_train, np.array(pd.get_dummies(y_train).values), x_test, np.array(pd.get_dummies(y_test).values) #标签y转onehot编码


# 训练
def train(nn,MAX_ITER,x_train,y_train,x_test,y_test):
    for i in range(MAX_ITER):#迭代次数
        print("epoch: ",i+1,"/",MAX_ITER)
        for j in range(x_train.shape[0]):
            # nn.inspect()
            nn.train(x_train[j],y_train[j])
        print("train set Cross_Entropy:",nn.Cross_Entropy(x_train,y_train))
        print("test set Cross_Entropy:", nn.Cross_Entropy(x_test, y_test))
        print("predict train:",nn.predict(x_train,y_train))# 验证
        print("predict test:",nn.predict(x_test, y_test) )# 验证


def main():
    # 使用数据集 train.csv 和test.csv
    x_train, y_train, x_test, y_test = read_data()
    nn = NeuralNetwork(2, 8, 3)  # 设置神经网络节点数和初始化参数
    MAX_ITER = 1000
    train(nn, MAX_ITER, x_train, y_train, x_test, y_test)

    #使用数据集_thyroid
    x_train, y_train, x_test, y_test = read_data_thyroid()
    nn = NeuralNetwork(21, 64, 3)  # 设置神经网络节点数和初始化参数
    MAX_ITER = 500
    train(nn, MAX_ITER, x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    main()
