import numpy as np


def compute_lie_gae(rewards, values, next_values_g, masks, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = np.zeros_like(rewards[0], dtype=np.float32)
    for step in reversed(range(rewards.shape[0])):
        delta = rewards[step] + gamma * masks[step + 1] * next_values_g[step] - values[step]
        gae = delta + gamma * gae_lambda * masks[step + 1] * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns
