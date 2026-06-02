from ppo.lie.groups.rn import RnGroup
from ppo.lie.groups.so2 import SO2Group
from ppo.lie.groups.se2 import SE2Group
from ppo.lie.groups.so3 import SO3Group
from ppo.lie.groups.se3 import SE3Group


def build_lie_group(name, algebra_dim=None):
    name = (name or "none").lower()
    if name in ("none", ""):
        return None
    if name in ("rn", "r", "translation"):
        if algebra_dim is None:
            raise ValueError("RnGroup requires algebra_dim.")
        return RnGroup(algebra_dim)
    if name == "so2":
        return SO2Group()
    if name == "se2":
        return SE2Group()
    if name == "so3":
        return SO3Group()
    if name == "se3":
        return SE3Group()
    raise ValueError(f"Unsupported Lie group '{name}'.")


__all__ = ["RnGroup", "SO2Group", "SE2Group", "SO3Group", "SE3Group", "build_lie_group"]
