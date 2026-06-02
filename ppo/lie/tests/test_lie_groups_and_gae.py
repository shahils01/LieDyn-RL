import numpy as np
import torch
from types import SimpleNamespace

from ppo.lie.gae import compute_lie_gae
from ppo.lie.dynamics import LieDynamics
from ppo.lie.groups.rn import RnGroup
from ppo.lie.groups.se2 import SE2Group
from ppo.lie.groups.se3 import SE3Group
from ppo.lie.groups.so2 import SO2Group
from ppo.lie.groups.so3 import SO3Group
from ppo.lie.losses import learned_lie_next_obs_with_z, lie_dynamics_losses
from ppo.lie.returns import lie_target_mix_alpha
from ppo.lie.state import LieStateSpec


def test_so2_exp_is_rotation():
    group = SO2Group()
    xi = torch.randn(128, 1)
    g = group.exp(xi)
    eye = torch.eye(2).expand(128, 2, 2)
    assert torch.allclose(g.transpose(-1, -2) @ g, eye, atol=1e-5)


def test_so2_log_exp_inverse_small_angles():
    group = SO2Group()
    xi = 0.1 * torch.randn(128, 1)
    xi_rec = group.log(group.exp(xi))
    assert torch.allclose(xi, xi_rec, atol=1e-5)


def test_rn_observed_transition_reconstructs_next_state():
    group = RnGroup(4)
    x = torch.randn(32, 4)
    x_next = torch.randn(32, 4)
    xi = group.infer_algebra(x, x_next)
    x_next_g = group.act(group.exp(xi), x)
    assert torch.allclose(x_next, x_next_g, atol=1e-6)


def test_se2_observed_transition_reconstructs_next_pose():
    group = SE2Group()
    x = torch.randn(32, 3)
    x_next = torch.randn(32, 3)
    x[:, 2] = 0.2 * x[:, 2]
    x_next[:, 2] = 0.2 * x_next[:, 2]
    xi = group.infer_algebra(x, x_next)
    x_next_g = group.act(group.exp(xi), x)
    assert torch.allclose(x_next, x_next_g, atol=1e-5)


def test_so3_log_exp_inverse_small_rotations():
    group = SO3Group()
    xi = 0.1 * torch.randn(64, 3)
    xi_rec = group.log(group.exp(xi))
    assert torch.allclose(xi, xi_rec, atol=1e-5)


def test_so3_observed_transition_reconstructs_next_quaternion():
    group = SO3Group()
    x = group.exp(0.2 * torch.randn(32, 3))
    delta = group.exp(0.1 * torch.randn(32, 3))
    x_next = group.act(delta, x)
    xi = group.infer_algebra(x, x_next)
    x_next_g = group.act(group.exp(xi), x)
    assert torch.allclose(x_next, x_next_g, atol=1e-5)


def test_se3_observed_transition_reconstructs_next_pose():
    group = SE3Group()
    x = torch.cat([torch.randn(32, 3), SO3Group().exp(0.2 * torch.randn(32, 3))], dim=-1)
    delta = torch.cat([torch.randn(32, 3), SO3Group().exp(0.1 * torch.randn(32, 3))], dim=-1)
    x_next = group.act(delta, x)
    xi = group.infer_algebra(x, x_next)
    x_next_g = group.act(group.exp(xi), x)
    assert torch.allclose(x_next[..., :3], x_next_g[..., :3], atol=1e-5)
    assert torch.allclose(x_next[..., 3:].abs(), x_next_g[..., 3:].abs(), atol=1e-5)


def test_lie_gae_zero_rewards_zero_values():
    rewards = np.zeros((8, 4, 1, 1), dtype=np.float32)
    values = np.zeros_like(rewards)
    next_values = np.zeros_like(rewards)
    masks = np.ones((9, 4, 1, 1), dtype=np.float32)
    adv, ret = compute_lie_gae(rewards, values, next_values, masks, 0.99, 0.95)
    assert np.allclose(adv, 0.0)
    assert np.allclose(ret, 0.0)


def test_lie_gae_matches_standard_gae_when_next_values_match():
    rng = np.random.default_rng(7)
    rewards = rng.normal(size=(6, 3, 1, 1)).astype(np.float32)
    values_all = rng.normal(size=(7, 3, 1, 1)).astype(np.float32)
    masks = np.ones((7, 3, 1, 1), dtype=np.float32)
    masks[4, 1, 0, 0] = 0.0

    lie_adv, lie_ret = compute_lie_gae(
        rewards=rewards,
        values=values_all[:-1],
        next_values_g=values_all[1:],
        masks=masks,
        gamma=0.98,
        gae_lambda=0.92,
    )

    standard_adv = np.zeros_like(rewards)
    gae = np.zeros_like(rewards[0])
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + 0.98 * values_all[step + 1] * masks[step + 1] - values_all[step]
        gae = delta + 0.98 * 0.92 * masks[step + 1] * gae
        standard_adv[step] = gae

    assert np.allclose(lie_adv, standard_adv)
    assert np.allclose(lie_ret, standard_adv + values_all[:-1])


def test_hybrid_alpha_schedule():
    args = SimpleNamespace(
        lie_mode="hybrid",
        lie_dynamics_warmup_updates=2,
        lie_dynamics_mix_updates=4,
    )
    policy = SimpleNamespace(args=args, lie_update_count=0)
    assert lie_target_mix_alpha(policy) == 0.0
    policy.lie_update_count = 4
    assert lie_target_mix_alpha(policy) == 0.5
    policy.lie_update_count = 10
    assert lie_target_mix_alpha(policy) == 1.0


def test_lie_dynamics_shape_and_learned_next_state_preserves_z():
    group = SE2Group()
    state_spec = LieStateSpec((5,), x_indices=np.array([0, 1, 2]))
    dynamics = LieDynamics(state_dim=5, action_dim=2, algebra_dim=group.algebra_dim, hidden_dim=16)
    obs = torch.randn(7, 5)
    next_obs = torch.randn(7, 5)
    actions = torch.randn(7, 2)
    next_obs_pred, xi = learned_lie_next_obs_with_z(group, state_spec, dynamics, obs, actions, next_obs)
    assert xi.shape == (7, 3)
    assert next_obs_pred.shape == next_obs.shape
    assert torch.allclose(next_obs_pred[:, 3:], next_obs[:, 3:])


def test_lie_dynamics_losses_are_finite():
    group = SO2Group()
    state_spec = LieStateSpec((4,), x_indices=np.array([0]))
    dynamics = LieDynamics(state_dim=4, action_dim=1, algebra_dim=group.algebra_dim, hidden_dim=16)
    obs = torch.randn(9, 4)
    next_obs = torch.randn(9, 4)
    actions = torch.randn(9, 1)
    dyn_loss, xi_loss, xi_norm = lie_dynamics_losses(group, state_spec, dynamics, obs, actions, next_obs)
    assert torch.isfinite(dyn_loss)
    assert torch.isfinite(xi_loss)
    assert torch.isfinite(xi_norm)
