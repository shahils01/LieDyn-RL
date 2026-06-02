from abc import ABC, abstractmethod

import torch


class LieGroup(ABC):
    @property
    @abstractmethod
    def algebra_dim(self):
        pass

    @abstractmethod
    def exp(self, xi):
        pass

    @abstractmethod
    def log(self, g):
        pass

    @abstractmethod
    def act(self, g, x):
        pass

    @abstractmethod
    def compose(self, g1, g2):
        pass

    @abstractmethod
    def inverse(self, g):
        pass

    def infer_algebra(self, x, x_next):
        raise NotImplementedError

    def distance(self, x1, x2):
        return torch.linalg.norm(x1 - x2, dim=-1, keepdim=True)

    def adjoint(self, g, xi):
        raise NotImplementedError


def wrap_angle(theta):
    return torch.atan2(torch.sin(theta), torch.cos(theta))
