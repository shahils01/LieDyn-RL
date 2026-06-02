import torch

from ppo.lie.groups.base import LieGroup, wrap_angle


class SE2Group(LieGroup):
    @property
    def algebra_dim(self):
        return 3

    def exp(self, xi):
        tx = xi[..., 0]
        ty = xi[..., 1]
        theta = xi[..., 2]
        c = torch.cos(theta)
        s = torch.sin(theta)
        zeros = torch.zeros_like(theta)
        ones = torch.ones_like(theta)
        row1 = torch.stack([c, -s, tx], dim=-1)
        row2 = torch.stack([s, c, ty], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def log(self, g):
        theta = torch.atan2(g[..., 1, 0], g[..., 0, 0])
        return torch.stack([g[..., 0, 2], g[..., 1, 2], theta], dim=-1)

    def act(self, g, x):
        if x.shape[-1] != 3:
            raise ValueError("SE2Group acts on planar poses [x, y, theta].")
        xy = x[..., :2]
        theta = x[..., 2:3]
        rot = g[..., :2, :2]
        trans = g[..., :2, 2]
        next_xy = torch.matmul(rot, xy.unsqueeze(-1)).squeeze(-1) + trans
        delta_theta = torch.atan2(g[..., 1, 0], g[..., 0, 0]).unsqueeze(-1)
        next_theta = wrap_angle(theta + delta_theta)
        return torch.cat([next_xy, next_theta], dim=-1)

    def compose(self, g1, g2):
        return torch.matmul(g1, g2)

    def inverse(self, g):
        rot_t = g[..., :2, :2].transpose(-1, -2)
        trans = -torch.matmul(rot_t, g[..., :2, 2].unsqueeze(-1)).squeeze(-1)
        zeros = torch.zeros_like(trans[..., 0])
        ones = torch.ones_like(trans[..., 0])
        row1 = torch.stack([rot_t[..., 0, 0], rot_t[..., 0, 1], trans[..., 0]], dim=-1)
        row2 = torch.stack([rot_t[..., 1, 0], rot_t[..., 1, 1], trans[..., 1]], dim=-1)
        row3 = torch.stack([zeros, zeros, ones], dim=-1)
        return torch.stack([row1, row2, row3], dim=-2)

    def infer_algebra(self, x, x_next):
        if x.shape[-1] != 3:
            raise ValueError("SE2Group requires x dimension 3.")
        dtheta = wrap_angle(x_next[..., 2:3] - x[..., 2:3])
        c = torch.cos(dtheta.squeeze(-1))
        s = torch.sin(dtheta.squeeze(-1))
        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        rot = torch.stack([row1, row2], dim=-2)
        rotated_xy = torch.matmul(rot, x[..., :2].unsqueeze(-1)).squeeze(-1)
        trans = x_next[..., :2] - rotated_xy
        return torch.cat([trans, dtheta], dim=-1)

    def distance(self, x1, x2):
        dxy = x2[..., :2] - x1[..., :2]
        dtheta = wrap_angle(x2[..., 2:3] - x1[..., 2:3])
        return torch.sqrt((dxy * dxy).sum(dim=-1, keepdim=True) + dtheta * dtheta)
