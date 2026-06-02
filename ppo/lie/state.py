import copy

import numpy as np
import torch


def parse_index_spec(spec, dim):
    if spec is None or spec == "":
        return None
    indices = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = part.split(":")
            if len(pieces) > 3:
                raise ValueError(f"Invalid index range '{part}'.")
            start = int(pieces[0]) if pieces[0] else 0
            stop = int(pieces[1]) if len(pieces) > 1 and pieces[1] else dim
            step = int(pieces[2]) if len(pieces) > 2 and pieces[2] else 1
            indices.extend(range(start, stop, step))
        else:
            indices.append(int(part))
    if not indices:
        return None
    normalized = [i + dim if i < 0 else i for i in indices]
    for idx in normalized:
        if idx < 0 or idx >= dim:
            raise ValueError(f"Index {idx} is out of bounds for vector dimension {dim}.")
    return np.asarray(normalized, dtype=np.int64)


class LieStateSpec:
    def __init__(self, obs_shape, x_indices=None, obs_key="policy"):
        self.obs_shape = obs_shape
        self.obs_key = obs_key
        self.obs_is_dict = isinstance(obs_shape, dict)
        vector_shape = obs_shape[obs_key] if self.obs_is_dict else obs_shape
        if len(vector_shape) != 1:
            raise ValueError("Lie-GAE currently requires a flat vector observation or a dict vector entry.")
        self.vector_dim = int(vector_shape[0])
        self.x_indices = np.asarray(x_indices, dtype=np.int64)
        self.x_dim = int(len(self.x_indices))

    @classmethod
    def from_args(cls, args, obs_shape):
        obs_key = getattr(args, "lie_obs_key", "policy")
        vector_shape = obs_shape[obs_key] if isinstance(obs_shape, dict) else obs_shape
        vector_dim = int(vector_shape[0])
        x_indices = parse_index_spec(getattr(args, "lie_x_indices", None), vector_dim)
        group_name = getattr(args, "lie_group", "rn").lower()
        if x_indices is None:
            if group_name in ("rn", "r", "translation"):
                x_indices = np.arange(vector_dim, dtype=np.int64)
            elif group_name == "so2":
                default_dim = 2 if vector_dim >= 2 else 1
                x_indices = np.arange(default_dim, dtype=np.int64)
            elif group_name == "se2" and vector_dim >= 3:
                x_indices = np.arange(3, dtype=np.int64)
            elif group_name == "so3" and vector_dim >= 4:
                x_indices = np.arange(4, dtype=np.int64)
            elif group_name == "se3" and vector_dim >= 7:
                x_indices = np.arange(7, dtype=np.int64)
            else:
                raise ValueError("--lie_x_indices is required for this Lie group and observation shape.")
        return cls(obs_shape, x_indices=x_indices, obs_key=obs_key)

    def vector(self, obs):
        return obs[self.obs_key] if self.obs_is_dict else obs

    def vector_torch(self, obs):
        return obs[self.obs_key] if self.obs_is_dict else obs

    def with_vector(self, obs, vector):
        if self.obs_is_dict:
            out = {k: np.array(v, copy=True) for k, v in obs.items()}
            out[self.obs_key] = vector
            return out
        return vector

    def extract_x_np(self, obs):
        return np.take(self.vector(obs), self.x_indices, axis=-1)

    def replace_x_np(self, obs, x):
        vector = np.array(self.vector(obs), copy=True)
        vector[..., self.x_indices] = x
        return self.with_vector(obs, vector)

    def extract_x_torch(self, obs):
        vector = obs[self.obs_key] if self.obs_is_dict else obs
        index = torch.as_tensor(self.x_indices, dtype=torch.long, device=vector.device)
        return torch.index_select(vector, dim=-1, index=index)

    def replace_x_torch(self, obs, x):
        vector = obs[self.obs_key] if self.obs_is_dict else obs
        next_vector = vector.clone()
        index = torch.as_tensor(self.x_indices, dtype=torch.long, device=vector.device)
        next_vector.index_copy_(-1, index, x)
        if self.obs_is_dict:
            out = copy.copy(obs)
            out[self.obs_key] = next_vector
            return out
        return next_vector
