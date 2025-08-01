from functools import reduce

class Perceptron(object):
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    # __str__是特殊方法，用于定义对象的字符串表示,其中%s是一个占位符，用于插入字符串，%f是一个占位符，用于插入浮点数
    # \n表示换行符
    def __str__(self):
        return 'weights\t:%s\nbias\t:%f\n' % (self.weights, self.bias)

    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a, b : a + b, 
                   map(lambda x_w: x_w[0] * x_w[1], zip(input_vec, self.weights)), 
                   0.0) + self.bias)

    def train(self, input_vecs, labels, iteration, rate):
        for i in range(iteration):
            self._one_iteration(input_vecs, labels, rate)
    def _one_iteration(self, input_vecs, labels, rate):
        samples = zip(input_vecs, labels)
        for(input_vec, label) in samples:
            output = self.predict(input_vec)
            self._update_weights(input_vec, output, label, rate)
    
    def _update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map(
            lambda x_w: x_w[1] + rate * delta * x_w[0],
            zip(input_vec, self.weights)))

        self.bias += rate*delta
    
    @staticmethod
    def f(x):
        return 1 if x > 0 else 0

    @staticmethod
    def get_training_dataset():
        input_vecs = [[1,1,1], [0,0,0], [1,0,0], [0,1,0]]
        labels = [1, 0, 0, 0]
        return input_vecs, labels
    
    @staticmethod
    def train_and_perceptron():
        p = Perceptron(3, Perceptron.f)
        input_vecs, labels = Perceptron.get_training_dataset()
        p.train(input_vecs, labels, 10, 0.1)
        return p


# 检查当前模块是否作为主程序运行
if __name__ == '__main__':
    and_perception = Perceptron.train_and_perceptron()
    print(and_perception)
    print('1 and 1 and 1 = %d' % and_perception.predict([1, 1, 1]))
    print('0 and 0 and 0 = %d' % and_perception.predict([0, 0, 0]))
    print('1 and 0 and 0 = %d' % and_perception.predict([1, 0, 0]))
    print('0 and 1 and 0 = %d' % and_perception.predict([0, 1, 0]))



