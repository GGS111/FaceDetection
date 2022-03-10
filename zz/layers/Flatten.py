from torch import nn

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, x):
        #print(x.shape)
        batch_size = x.shape[0]
        return x.reshape(batch_size, -1)
