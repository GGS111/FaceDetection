from .Layer import Layer

class Input(Layer):
    def __init__(self, shape):
        super(Input, self).__init__()
        self.shape = (-1,) + tuple(shape)
    def forward(self, x):
        return x.reshape(self.shape);
