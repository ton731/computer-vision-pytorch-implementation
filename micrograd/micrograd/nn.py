import random
from micrograd.engine import Value


class Module:

    def parameters(self):
        return []



class Neuron(Module):
    
    def __init__(self, n_in, act=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_in)]
        self.b = Value(random.uniform(-1, 1))
        self.act = act

    def __repr__(self):
        return f"{'ReLU' if self.act else 'Linear'}Neuron({len(self.w)})"
        
    def __call__(self, x):
        # w * x + b
        out = sum([wi * xi for wi, xi in zip(self.w, x)]) + self.b
        return out.relu() if self.act else out
    
    def parameters(self):
        return self.w + [self.b]
    


class Layer(Module):
    
    def __init__(self, n_in, n_out, **kwargs):
        self.neurons = [Neuron(n_in, **kwargs) for _ in range(n_out)]

    def __repr__(self):
        return f"Layer: [{', '.join(str(n) for n in self.neurons)}]"
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
    


class MLP(Module):
    
    def __init__(self, n_in, n_outs, **kwargs):
        sz = [n_in] + n_outs
        self.layers = [Layer(sz[i], sz[i+1], **kwargs) for i in range(len(n_outs))]

    def __repr__(self):
        return f"MLP: [{', '.join(str(layer) for layer in self.layers)}]"
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    