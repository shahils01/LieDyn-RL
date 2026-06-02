import torch

from ppo.lie.groups.base import LieGroup, wrap_angle


class SO2Group(LieGroup):
    @property
    def algebra_dim(self):
        return 1

    def exp(self, xi):
        theta = xi.squeeze(-1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def log(self, g):
        return torch.atan2(g[..., 1, 0], g[..., 0, 0]).unsqueeze(-1)

    def act(self, g, x):
        if x.shape[-1] == 1:
            theta = self.log(g)
            return wrap_angle(x + theta)
        if x.shape[-1] == 2:
            return torch.matmul(g, x.unsqueeze(-1)).squeeze(-1)
        raise ValueError("SO2Group acts on angle states [theta] or orientation vectors [cos, sin].")

    def compose(self, g1, g2):
        return torch.matmul(g1, g2)

    def inverse(self, g):
        return g.transpose(-1, -2)

    def infer_algebra(self, x, x_next):
        if x.shape[-1] == 1:
            return wrap_angle(x_next - x)
        if x.shape[-1] == 2:
            angle = torch.atan2(x[..., 1], x[..., 0])
            next_angle = torch.atan2(x_next[..., 1], x_next[..., 0])
            return wrap_angle(next_angle - angle).unsqueeze(-1)
        raise ValueError("SO2Group requires x dimension 1 or 2.")

    def distance(self, x1, x2):
        if x1.shape[-1] == 1:
            return torch.abs(wrap_angle(x2 - x1))
        return torch.linalg.norm(x1 - x2, dim=-1, keepdim=True)
