class StandardScaler:
    """Standard scaler with epsilon guard for zero std."""

    def __init__(self, mean, std, eps=1e-6):
        self.mean = mean
        self.std = std if abs(std) > eps else eps
        self.eps = eps

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean