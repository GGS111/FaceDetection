class WrappedDataLoader:
    def __init__(self, dl, func=lambda *x: x):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for _b in batches:
            yield (self.func(*_b))
