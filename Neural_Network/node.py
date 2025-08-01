from functools import reduce
import math

# 节点类，负责记录和维护节点自身信息以及与这个节点相关的上下游连接，实现输出值和误差项的计算
class Node(object):
    def __init__(self, layer_index, node_index):
        # 节点所属的层
        self.layer_index = layer_index
        # 节点在层中的位置
        self.node_index = node_index
        # 节点的下游节点
        self.downstream = []
        # 节点的上游节点
        self.upstream = []
        
        self.output = 0
        self.delta = 0
    
    # 添加一个下游连接
    def append_downstream_connection(self, conn):
        self.downstream.append(conn)
    # 添加一个上游连接
    def append_upstream_connection(self, conn):
        self.upstream.append(conn)
    
    # 设置节点的输出值，如果节点属于输入层，则输出值等于输入值，否则等于上一层所有节点的加权输出值
    def set_output(self, output):
        self.output = output

    # 计算节点加权输出值
    # self.upstream是上游节点列表，reduce函数将上游节点列表中的每个节点与连接的权重相乘，然后求和,ret在这里起到一个累积和的作用
    # 0.0是初始值，表示从0开始累加
    # conn.upstream_node.output 表示上游节点的输出值 作为输入值 与连接的权重相乘
    # conn.weight 表示连接的权重
    def calc_output(self):
        output = reduce(lambda ret, conn: ret + conn.upstream_node.output * conn.weight, self.upstream, 0.0)
        self.output = 1.0 / (1.0 + math.exp(-output))
        return self.output

    # 计算隐藏层节点的误差项
    def calc_hidden_layer_delta(self):
        downstream_delata = reduce(lambda ret, conn: ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output*(1-self.output)*downstream_delata
    
    # 计算输出层节点的误差项
    def calc_output_layer_delta(self, label):
        self.delta = self.output*(1 - self.output)*(label - self.output)

    # upstream和downstream是两个列表
    def __str__(self):
        '''
        打印节点信息
        '''
        node_str = '%u-%u: output:%f delta:%f' % (self.layer_index, self.node_index, self.output, self.delta)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        upstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.upstream, '')
        return node_str + '\n\tdownstream:' + downstream_str + '\n\tupstream:' + upstream_str

'''
常量节点,为了计算方便,在输出层之前添加一个节点,这个节点的输出值永远为1
'''
class ConstNode(object):
    def __init__(self, layer_index, node_index):
        '''
        构造函数
        '''
        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1.0

    def append_downstream_connection(self, conn):
        '''
        添加一个到下游节点的连接
        '''
        self.downstream.append(conn)
    
    def clac_hidden_layer_delta(self):
        '''
        计算隐藏层节点的误差项
        '''
        downstream_delta = reduce(lambda ret, conn : ret + conn.downstream_node.delta * conn.weight, self.downstream, 0.0)
        self.delta = self.output * (1 - self.output) * downstream_delta

    def __str__(self):
        '''
        打印节点信息
        '''
        node_str = '%u-%u: output:%f' % (self.layer_index, self.node_index, self.output)
        downstream_str = reduce(lambda ret, conn: ret + '\n\t' + str(conn), self.downstream, '')
        return node_str + '\n\tdownstream:' + downstream_str