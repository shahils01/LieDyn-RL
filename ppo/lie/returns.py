import numpy as np
import torch

from ppo.lie.gae import compute_lie_gae
from ppo.lie.losses import learned_lie_next_obs_with_z


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


def _to_tensor_obs(obs, device):
    if isinstance(obs, dict):
        return {k: torch.as_tensor(v, dtype=torch.float32, device=device) for k, v in obs.items()}
    return torch.as_tensor(obs, dtype=torch.float32, device=device)


def _to_numpy_obs(obs):
    if isinstance(obs, dict):
        return {k: v.detach().cpu().numpy() for k, v in obs.items()}
    return obs.detach().cpu().numpy()


def build_learned_lie_next_obs(buffer, policy, group, state_spec):
    obs_t = _flatten_time_env(buffer.obs[:-1])
    obs_next = _flatten_time_env(buffer.obs[1:])
    actions = buffer.actions.reshape(-1, *buffer.actions.shape[2:])

    device = policy.device
    obs_t = _to_tensor_obs(obs_t, device)
    obs_next = _to_tensor_obs(obs_next, device)
    actions = torch.as_tensor(actions, dtype=torch.float32, device=device).reshape(actions.shape[0], -1)
    with torch.no_grad():
        next_obs_g, _ = learned_lie_next_obs_with_z(
            group, state_spec, policy.transformer.lie_dynamics, obs_t, actions, obs_next
        )
    next_obs_g = _to_numpy_obs(next_obs_g)
    if isinstance(next_obs_g, dict):
        return {
            k: v.reshape(buffer.episode_length, buffer.n_rollout_threads, 1, *buffer.obs[k].shape[3:])
            for k, v in next_obs_g.items()
        }
    return next_obs_g.reshape(buffer.episode_length, buffer.n_rollout_threads, 1, *buffer.obs.shape[3:])


def lie_target_mix_alpha(policy):
    args = policy.args
    if getattr(args, "lie_mode", "observed") == "observed":
        return 0.0
    if getattr(args, "lie_mode", "observed") == "learned":
        return 1.0
    step = max(0, policy.lie_update_count - args.lie_dynamics_warmup_updates)
    if args.lie_dynamics_mix_updates <= 0:
        return 1.0
    return float(min(1.0, step / float(args.lie_dynamics_mix_updates)))


def compute_lie_returns(buffer, policy, group, state_spec, value_normalizer=None):
    if buffer.num_quants != 1:
        raise ValueError("Lie-GAE currently supports scalar critics only. Run with --num_quants 1.")

    next_obs_g = build_observed_lie_next_obs(buffer, group, state_spec)
    flat_next_obs_g = _flatten_time_env(next_obs_g)
    observed_next_values_g = policy.get_values(flat_next_obs_g, None).detach().cpu().numpy()
    observed_next_values_g = _unflatten_values(observed_next_values_g, buffer.episode_length, buffer.n_rollout_threads)

    alpha = lie_target_mix_alpha(policy)
    if alpha > 0.0:
        if policy.transformer.lie_dynamics is None:
            raise ValueError("Learned Lie targets require --lie_mode hybrid or --lie_mode learned.")
        learned_next_obs_g = build_learned_lie_next_obs(buffer, policy, group, state_spec)
        flat_learned_next_obs_g = _flatten_time_env(learned_next_obs_g)
        learned_next_values_g = policy.get_values(flat_learned_next_obs_g, None).detach().cpu().numpy()
        learned_next_values_g = _unflatten_values(learned_next_values_g, buffer.episode_length, buffer.n_rollout_threads)
        next_values_g = (1.0 - alpha) * observed_next_values_g + alpha * learned_next_values_g
    else:
        next_values_g = observed_next_values_g

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
    policy.last_lie_target_mix_alpha = alpha
