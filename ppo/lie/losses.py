import torch


def observed_algebra(group, state_spec, obs, next_obs):
    x = state_spec.extract_x_torch(obs)
    x_next = state_spec.extract_x_torch(next_obs)
    return group.infer_algebra(x, x_next)


def learned_lie_next_obs(group, state_spec, dynamics, obs, actions):
    x = state_spec.extract_x_torch(obs)
    xi = dynamics(state_spec.vector_torch(obs).float(), actions.float())
    x_next = group.act(group.exp(xi), x)
    return state_spec.replace_x_torch(obs, x_next), xi


def learned_lie_next_obs_with_z(group, state_spec, dynamics, obs, actions, next_obs):
    x = state_spec.extract_x_torch(obs)
    xi = dynamics(state_spec.vector_torch(obs).float(), actions.float())
    x_next = group.act(group.exp(xi), x)
    return state_spec.replace_x_torch(next_obs, x_next), xi


def lie_dynamics_losses(group, state_spec, dynamics, obs, actions, next_obs):
    x_next_obs = state_spec.extract_x_torch(next_obs)
    next_obs_pred, xi_pred = learned_lie_next_obs_with_z(group, state_spec, dynamics, obs, actions, next_obs)
    x_next_pred = state_spec.extract_x_torch(next_obs_pred)
    xi_obs = observed_algebra(group, state_spec, obs, next_obs).detach()

    dyn_loss = group.distance(x_next_obs, x_next_pred).pow(2).mean()
    xi_loss = (xi_pred - xi_obs).pow(2).mean()
    xi_norm = torch.linalg.norm(xi_pred.detach(), dim=-1).mean()
    return dyn_loss, xi_loss, xi_norm
