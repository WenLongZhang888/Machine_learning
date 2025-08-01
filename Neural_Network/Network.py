from Layer import Layer
from Connection import Connections
class Network(object):
    def __init__(self, layers):
        '''
        初始化一个全连接神经网络
        layers: 二维数组，描述神经网络每层节点数
        '''
        self.connections = Connections()
        self.layers = []
        for i in range(len(layers) - 1):
            self.layers.append(Layer(i, layers[i], layers[i+1]))
    
    def train(self, labels, data_set, rate, iteration):
        '''
        训练神经网络
        labels: 数组，训练样本标签
        data_set: 二维数组，训练样本特征
        rate: 学习率
        iteration: 迭代次数
        '''
        for i in range(iteration):
            for d in range(len(data_set)):
                self._train_one_sample(labels[d], data_set[d], rate)
    
    def train_one_sample(self, label, sample, rate):
        '''
        训练一个样本
        label: 样本标签
        sample: 样本特征
        rate: 学习率
        '''
        self.predict(sample)
        self._calc_delta(label)
        self._update_weight(rate)


    # range(start, stop, step) 函数返回一个包含从start到stop的整数，步长为step的列表，不包括stop
    def predict(self, sample):
        '''
        根据输入的样本预测输出值
        '''

        self.layers[0].set_output(sample)
        for i in range(1, len(self.layers)):
            self.layers[i].calc_output()
        # self.layers[-1]表示网络的最后一层
        # self.layers[-1].nodes[:-1]表示网络的最后一层节点，不包括输出层最后一个节点
        # 将 lambda 函数应用于 self.layers[-1].nodes[:-1] 中的每个节点对象。
        # 也就是说，它会遍历最后一层中除了最后一个节点之外的所有节点，并提取每个节点的 output 属性
        return map(lambda node: node.output, self.layers[-1].nodes[:-1])



    def calc_delta(self, label):
        '''
        计算每个节点的delta
        '''
        # self.layers[-1]表示网络的最后一层
        # 遍历输出节点，排除输出层最后一个节点
        output_nodes = self.layers[-1].nodes
        for i in range(len(output_nodes) - 1):
            output_nodes[i].calc_output_layer_delta(label)
        # 遍历除去输出层之外的各层节点，根据上一层的节点输出计算delta
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self, rate):
        '''
        更新每个连接权重
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.connections:
                    conn.update_weight(rate)
    
    def calc_gradient(self):
        '''
        计算每个连接上的梯度
        '''
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.connections:
                    conn.calc_gradient()
    
    def get_gradient(self, label, sample):
        '''
        获得网络在一个样本下，每个连接上的梯度
        '''
        self.predict(sample)
        self.calc_delta(label)
        self.calc_gradient()
    
    def dump(self):
        '''
        打印网络信息
        '''
        for layer in self.layers:
            layer.dump()