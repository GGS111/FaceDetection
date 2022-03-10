class History:
    def __init__(self):
        self.epoch = []
        self.history = {}
        
    def add_epoch_values(self, epoch, logs=None):
        logs = logs or {}
        self.epoch.append(epoch)
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
