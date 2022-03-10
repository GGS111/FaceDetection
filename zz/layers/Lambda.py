from .Layer import Layer

class Lambda(Layer):
    def __init__(self, lambd):
        super(Lambda, self).__init__()
        self.lambd = lambd
    def forward(self, *x):
        return self.lambd(*x)
