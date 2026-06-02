from ppo.lie.groups.base import LieGroup


class RnGroup(LieGroup):
    def __init__(self, dim):
        self._algebra_dim = int(dim)

    @property
    def algebra_dim(self):
        return self._algebra_dim

    def exp(self, xi):
        return xi

    def log(self, g):
        return g

    def act(self, g, x):
        return x + g

    def compose(self, g1, g2):
        return g1 + g2

    def inverse(self, g):
        return -g

    def infer_algebra(self, x, x_next):
        return x_next - x
