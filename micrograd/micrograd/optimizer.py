"""Implementation of optimizers"""

class BaseOptimizer:

    def __init__(self, parameters, learning_rate):
        self.parameters = parameters
        self.learning_rate = learning_rate

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0

    def step(self):
        raise NotImplementedError



class SGD(BaseOptimizer):

    def __init__(self, parameters, learning_rate=0.1, decay_rate=0.01):
        super().__init__(parameters, learning_rate)
        self.decay_rate = decay_rate
        self.count = 0

    def __repr__(self):
        return f"SGD Optimizer, num_params={len(self.parameters)}, lr={self.learning_rate}, decay_rate={self.decay_rate}"

    def lr(self):
        return max(1e-3, self.learning_rate * (1 - self.count * self.decay_rate))

    def step(self):
        lr = self.lr()
        for p in self.parameters:
            p.data -= lr * p.grad
        self.count += 1


