import torch

from ppo.lie.groups.base import LieGroup
from ppo.lie.groups.so3 import (
    SO3Group,
    _quat_inv,
    _quat_mul,
    _quat_normalize,
    _quat_standardize,
    _skew,
    _quat_to_matrix,
)


class SE3Group(LieGroup):
    def __init__(self):
        self.so3 = SO3Group()

    @property
    def algebra_dim(self):
        return 6

    def _eye(self, xi):
        eye = torch.eye(3, dtype=xi.dtype, device=xi.device)
        return eye.expand(*xi.shape[:-1], 3, 3)

    def _left_jacobian_so3(self, omega):
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
        omega_hat = _skew(omega)
        omega_hat2 = torch.matmul(omega_hat, omega_hat)
        theta2 = theta * theta
        theta3 = theta2 * theta
        a = torch.where(theta > 1e-6, (1.0 - torch.cos(theta)) / theta2, 0.5 - theta2 / 24.0)
        b = torch.where(theta > 1e-6, (theta - torch.sin(theta)) / theta3, 1.0 / 6.0 - theta2 / 120.0)
        return self._eye(omega) + a.unsqueeze(-1) * omega_hat + b.unsqueeze(-1) * omega_hat2

    def _left_jacobian_so3_inv(self, omega):
        theta = torch.linalg.norm(omega, dim=-1, keepdim=True)
        omega_hat = _skew(omega)
        omega_hat2 = torch.matmul(omega_hat, omega_hat)
        theta2 = theta * theta
        half_theta = 0.5 * theta
        coef = torch.where(
            theta > 1e-6,
            (1.0 / theta2) - (1.0 + torch.cos(theta)) / (2.0 * theta * torch.sin(theta)).clamp_min(1e-8),
            1.0 / 12.0 + theta2 / 720.0,
        )
        return self._eye(omega) - 0.5 * omega_hat + coef.unsqueeze(-1) * omega_hat2

    def exp(self, xi):
        rho = xi[..., :3]
        omega = xi[..., 3:]
        rot = self.so3.exp(omega)
        trans = torch.matmul(self._left_jacobian_so3(omega), rho.unsqueeze(-1)).squeeze(-1)
        return torch.cat([trans, rot], dim=-1)

    def log(self, g):
        omega = self.so3.log(g[..., 3:])
        rho = torch.matmul(self._left_jacobian_so3_inv(omega), g[..., :3].unsqueeze(-1)).squeeze(-1)
        return torch.cat([rho, omega], dim=-1)

    def act(self, g, x):
        if x.shape[-1] != 7:
            raise ValueError("SE3Group acts on poses [x, y, z, qw, qx, qy, qz].")
        trans = g[..., :3]
        rot = _quat_normalize(g[..., 3:])
        pos = x[..., :3]
        quat = _quat_normalize(x[..., 3:])
        rot_mat = _quat_to_matrix(rot)
        next_pos = torch.matmul(rot_mat, pos.unsqueeze(-1)).squeeze(-1) + trans
        next_quat = _quat_standardize(_quat_normalize(_quat_mul(rot, quat)))
        return torch.cat([next_pos, next_quat], dim=-1)

    def compose(self, g1, g2):
        rot1 = _quat_normalize(g1[..., 3:])
        rot2 = _quat_normalize(g2[..., 3:])
        trans = g1[..., :3] + torch.matmul(_quat_to_matrix(rot1), g2[..., :3].unsqueeze(-1)).squeeze(-1)
        rot = _quat_standardize(_quat_normalize(_quat_mul(rot1, rot2)))
        return torch.cat([trans, rot], dim=-1)

    def inverse(self, g):
        rot = _quat_normalize(g[..., 3:])
        inv_rot = _quat_inv(rot)
        inv_trans = -torch.matmul(_quat_to_matrix(inv_rot), g[..., :3].unsqueeze(-1)).squeeze(-1)
        return torch.cat([inv_trans, inv_rot], dim=-1)

    def infer_algebra(self, x, x_next):
        if x.shape[-1] != 7:
            raise ValueError("SE3Group requires pose states [x, y, z, qw, qx, qy, qz].")
        rot_delta = self.so3.compose(x_next[..., 3:], self.so3.inverse(x[..., 3:]))
        rot_mat = _quat_to_matrix(rot_delta)
        trans = x_next[..., :3] - torch.matmul(rot_mat, x[..., :3].unsqueeze(-1)).squeeze(-1)
        return torch.cat([trans, self.so3.log(rot_delta)], dim=-1)

    def distance(self, x1, x2):
        dpos = x2[..., :3] - x1[..., :3]
        drot = self.so3.distance(x1[..., 3:], x2[..., 3:])
        return torch.sqrt((dpos * dpos).sum(dim=-1, keepdim=True) + drot * drot)
