import numpy as np
import torch

from ppo.lie.gae import compute_lie_gae


def _flatten_time_env(obs):
    if isinstance(obs, dict):
        return {k: v.reshape(-1, *v.shape[3:]) for k, v in obs.items()}
    return obs.reshape(-1, *obs.shape[3:])


def _unflatten_values(values, episode_length, n_rollout_threads):
    return values.reshape(episode_length, n_rollout_threads, 1, -1)


def build_observed_lie_next_obs(buffer, group, state_spec):
    obs_t = buffer.obs[:-1]
    obs_next = buffer.obs[1:]
    x = state_spec.extract_x_np(obs_t)
    x_next = state_spec.extract_x_np(obs_next)

    device = torch.device("cpu")
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    x_next_t = torch.as_tensor(x_next, dtype=torch.float32, device=device)
    with torch.no_grad():
        xi = group.infer_algebra(x_t, x_next_t)
        x_next_g = group.act(group.exp(xi), x_t).cpu().numpy()
    return state_spec.replace_x_np(obs_next, x_next_g)


def compute_lie_returns(buffer, policy, group, state_spec, value_normalizer=None):
    if buffer.num_quants != 1:
        raise ValueError("Lie-GAE currently supports scalar critics only. Run with --num_quants 1.")

    next_obs_g = build_observed_lie_next_obs(buffer, group, state_spec)
    flat_next_obs_g = _flatten_time_env(next_obs_g)
    next_values_g = policy.get_values(flat_next_obs_g, None).detach().cpu().numpy()
    next_values_g = _unflatten_values(next_values_g, buffer.episode_length, buffer.n_rollout_threads)

    values = buffer.value_preds[:-1]
    if value_normalizer is not None:
        values = value_normalizer.denormalize(values)
        next_values_g = value_normalizer.denormalize(next_values_g)

    advantages, returns = compute_lie_gae(
        rewards=buffer.rewards,
        values=values,
        next_values_g=next_values_g,
        masks=buffer.masks,
        gamma=buffer.gamma,
        gae_lambda=buffer.gae_lambda,
    )
    buffer.advantages[...] = advantages
    buffer.returns[:-1] = returns
