import torch

from ppo.lie.groups.base import LieGroup


def _skew(v):
    zeros = torch.zeros_like(v[..., 0])
    row1 = torch.stack([zeros, -v[..., 2], v[..., 1]], dim=-1)
    row2 = torch.stack([v[..., 2], zeros, -v[..., 0]], dim=-1)
    row3 = torch.stack([-v[..., 1], v[..., 0], zeros], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def _quat_normalize(q):
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_standardize(q):
    sign = torch.where(q[..., :1] < 0.0, -1.0, 1.0)
    return q * sign


def _quat_mul(q1, q2):
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def _quat_inv(q):
    q = _quat_normalize(q)
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_to_matrix(q):
    q = _quat_normalize(q)
    w, x, y, z = q.unbind(dim=-1)
    ww, xx, yy, zz = w * w, x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z
    row1 = torch.stack([ww + xx - yy - zz, 2 * (xy - wz), 2 * (xz + wy)], dim=-1)
    row2 = torch.stack([2 * (xy + wz), ww - xx + yy - zz, 2 * (yz - wx)], dim=-1)
    row3 = torch.stack([2 * (xz - wy), 2 * (yz + wx), ww - xx - yy + zz], dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def _matrix_to_quat(m):
    trace = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    qw = 0.5 * torch.sqrt((1.0 + trace).clamp_min(1e-8))
    qx = (m[..., 2, 1] - m[..., 1, 2]) / (4.0 * qw).clamp_min(1e-8)
    qy = (m[..., 0, 2] - m[..., 2, 0]) / (4.0 * qw).clamp_min(1e-8)
    qz = (m[..., 1, 0] - m[..., 0, 1]) / (4.0 * qw).clamp_min(1e-8)
    return _quat_standardize(_quat_normalize(torch.stack([qw, qx, qy, qz], dim=-1)))


class SO3Group(LieGroup):
    @property
    def algebra_dim(self):
        return 3

    def exp(self, xi):
        theta = torch.linalg.norm(xi, dim=-1, keepdim=True)
        half = 0.5 * theta
        scale = torch.where(theta > 1e-8, torch.sin(half) / theta, 0.5 - theta * theta / 48.0)
        return _quat_standardize(_quat_normalize(torch.cat([torch.cos(half), scale * xi], dim=-1)))

    def log(self, g):
        q = _quat_standardize(_quat_normalize(g))
        v = q[..., 1:]
        v_norm = torch.linalg.norm(v, dim=-1, keepdim=True)
        angle = 2.0 * torch.atan2(v_norm, q[..., :1].clamp_min(1e-8))
        scale = torch.where(v_norm > 1e-8, angle / v_norm, 2.0 + angle * angle / 12.0)
        return scale * v

    def act(self, g, x):
        if x.shape[-1] != 4:
            raise ValueError("SO3Group acts on unit quaternions [w, x, y, z].")
        return _quat_standardize(_quat_normalize(_quat_mul(g, x)))

    def compose(self, g1, g2):
        return _quat_standardize(_quat_normalize(_quat_mul(g1, g2)))

    def inverse(self, g):
        return _quat_inv(g)

    def infer_algebra(self, x, x_next):
        if x.shape[-1] != 4:
            raise ValueError("SO3Group requires quaternion states [w, x, y, z].")
        return self.log(self.compose(x_next, self.inverse(x)))

    def distance(self, x1, x2):
        return torch.linalg.norm(self.infer_algebra(x1, x2), dim=-1, keepdim=True)


__all__ = [
    "SO3Group",
    "_quat_normalize",
    "_quat_standardize",
    "_quat_mul",
    "_quat_inv",
    "_quat_to_matrix",
    "_matrix_to_quat",
    "_skew",
]
