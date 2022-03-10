from .Layer import Layer

class Reshape(Layer):
    def __init__(self, target_shape, **kwargs):
        super(Reshape, self).__init__(**kwargs)
        self.target_shape = tuple(target_shape)
    def forward(self, x):
        new_shape = (x.shape[0],) + self.target_shape
        return x.reshape(new_shape)

