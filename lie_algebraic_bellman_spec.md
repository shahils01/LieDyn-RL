# Lie-Algebraic Bellman Equation, Advantage Function, and GAE

This document is a coding-oriented specification for implementing a geometry-aware actor-critic method in which the transition of the geometric state is represented by a Lie-algebra element.

The central idea is:

```math
s_t = (x_t, z_t), \qquad x_t \in \mathcal M,
```

where `x_t` is the geometric part of the state and `z_t` contains non-geometric variables such as velocity, contacts, object parameters, proprioception, or latent context.

A Lie group `G` acts on the geometric state manifold `M`:

```math
G \times \mathcal M \to \mathcal M,
\qquad
(g,x) \mapsto g \cdot x.
```

A transition is represented through a Lie-algebra displacement:

```math
\xi_t \in \mathfrak g,
\qquad
g_t = \exp(\xi_t) \in G,
```

and

```math
x_{t+1} = \exp(\xi_t) \cdot x_t.
```

The transition model becomes

```math
s_{t+1}^G = F^G(s_t,a_t)
= \left(\exp(\xi_t)\cdot x_t, z_{t+1}\right).
```

In the learned implicit-dynamics version:

```math
\xi_t = \xi_\theta(s_t,a_t),
```

so

```math
\hat s_{t+1}^G
= F_\theta^G(s_t,a_t)
= \left(\exp(\xi_\theta(s_t,a_t))\cdot x_t, \hat z_{t+1}\right).
```

---

## 1. Goals for implementation

A first implementation should support:

1. A Lie group abstraction with `exp`, `log`, `act`, `compose`, `inverse`, and optionally `adjoint`.
2. A value model `V_phi(s)`.
3. A policy model `pi_psi(a | s)`.
4. An optional Lie dynamics model `xi_theta(s,a)`.
5. A Lie-geometric TD residual.
6. Lie-Algebraic GAE.
7. Actor-critic training, preferably PPO-compatible.

The minimum useful prototype can use observed transitions and infer `xi_t` from data. A more ambitious version learns `xi_theta(s,a)` and uses it inside the Bellman target.

---

## 2. Notation

State decomposition:

```math
s = (x,z),
```

where:

```math
x \in \mathcal M,
\qquad
z \in \mathcal Z.
```

Lie group and Lie algebra:

```math
G: \text{Lie group},
\qquad
\mathfrak g = T_eG.
```

Group action:

```math
x' = g \cdot x.
```

Lie algebra transition:

```math
x' = \exp(\xi)\cdot x,
\qquad
\xi \in \mathfrak g.
```

Discount:

```math
\gamma \in [0,1).
```

GAE parameter:

```math
\lambda \in [0,1].
```

---

## 3. Lie-Algebraic Bellman equation

The ordinary policy Bellman equation is

```math
V^\pi(s)
=
\mathbb E_{a\sim\pi(\cdot|s),\,s'\sim P(\cdot|s,a)}
\left[
 r(s,a,s') + \gamma V^\pi(s')
\right].
```

With Lie-geometric dynamics,

```math
s' = \left(\exp(\xi)\cdot x, z'\right),
```

so the Bellman equation becomes

```math
V^\pi(x,z)
=
\mathbb E_{a\sim\pi(\cdot|x,z)}
\mathbb E_{\xi,z'\sim q(\cdot|x,z,a)}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma V^\pi\left(\exp(\xi)\cdot x,z'\right)
\right].
```

Define the Lie-geometric Bellman operator:

```math
\mathcal T_\pi^G V(x,z)
=
\mathbb E_{a\sim\pi}
\mathbb E_{\xi,z'}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma V\left(\exp(\xi)\cdot x,z'\right)
\right].
```

Then

```math
V^\pi = \mathcal T_\pi^G V^\pi.
```

Optimal value version:

```math
V^*(x,z)
=
\sup_a
\mathbb E_{\xi,z'\sim q(\cdot|x,z,a)}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma V^*\left(\exp(\xi)\cdot x,z'\right)
\right].
```

---

## 4. Operator form using Lie derivatives

For an element

```math
\eta \in \mathfrak g,
```

define the fundamental vector field on `M`:

```math
\eta_{\mathcal M}(x)
=
\left.
\frac{d}{d\epsilon}
\right|_{\epsilon=0}
\exp(\epsilon\eta)\cdot x.
```

The Lie derivative of `V` along `eta` is

```math
\mathcal L_\eta V(x)
=
dV_x[\eta_{\mathcal M}(x)]
=
\left.
\frac{d}{d\epsilon}
\right|_{\epsilon=0}
V(\exp(\epsilon\eta)\cdot x).
```

For a finite algebra element `xi`, value transport along the group action is

```math
V(\exp(\xi)\cdot x)
=
\left(e^{\mathcal L_\xi}V\right)(x),
```

with

```math
e^{\mathcal L_\xi}
=
I + \mathcal L_\xi
+ \frac{1}{2}\mathcal L_\xi^2
+ \frac{1}{3!}\mathcal L_\xi^3
+ \cdots.
```

Ignoring `z` for a moment, the Lie-Algebraic Bellman equation can be written as

```math
V^\pi(x)
=
\mathbb E_{a\sim\pi}
\mathbb E_{\xi\sim q(\cdot|x,a)}
\left[
 r(x,a,\xi)
 +
 \gamma \left(e^{\mathcal L_\xi}V^\pi\right)(x)
\right].
```

Equivalently,

```math
(I-\gamma)V^\pi(x)
=
\mathbb E_{a,\xi}
\left[
 r(x,a,\xi)
 +
 \gamma
 \left(
   \mathcal L_\xi
   + \frac{1}{2}\mathcal L_\xi^2
   + \cdots
 \right)V^\pi(x)
\right].
```

This expresses Bellman backup as value transport along Lie-algebra directions.

---

## 5. Lie-Algebraic Q-function

The Q-function is

```math
Q^\pi(s,a)
=
\mathbb E
\left[
 r(s,a,s') + \gamma V^\pi(s') \mid s,a
\right].
```

With Lie dynamics:

```math
Q^\pi(x,z,a)
=
\mathbb E_{\xi,z'\sim q(\cdot|x,z,a)}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma V^\pi\left(\exp(\xi)\cdot x,z'\right)
\right].
```

And

```math
V^\pi(x,z)
=
\mathbb E_{a\sim\pi(\cdot|x,z)}[Q^\pi(x,z,a)].
```

The recursive Q equation is

```math
Q^\pi(x,z,a)
=
\mathbb E_{\xi,z'}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma
 \mathbb E_{a'\sim\pi(\cdot|s')}
 Q^\pi(s',a')
\right],
```

where

```math
s' = \left(\exp(\xi)\cdot x,z'\right).
```

---

## 6. Lie-Algebraic advantage function

The standard advantage is

```math
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s).
```

The Lie-Algebraic advantage is

```math
A_G^\pi(x,z,a)
=
\mathbb E_{\xi,z'}
\left[
 r(x,z,a,\xi,z')
 +
 \gamma V^\pi\left(\exp(\xi)\cdot x,z'\right)
 -
 V^\pi(x,z)
\right].
```

Define the one-step Lie-geometric TD residual:

```math
\delta_t^G
=
r_t
+
\gamma V_\phi\left(\exp(\xi_t)\cdot x_t,z_{t+1}\right)
-
V_\phi(x_t,z_t).
```

If `V_phi = V^pi` and the Lie transition distribution is correct, then

```math
A_G^\pi(s_t,a_t)
=
\mathbb E[\delta_t^G \mid s_t,a_t].
```

---

## 7. Local approximation of the advantage

For small `xi`, expand value locally:

```math
V(\exp(\xi)\cdot x,z)
=
V(x,z)
+
\langle D_{\mathfrak g}V(x,z),\xi\rangle
+
\frac{1}{2}\langle \xi,H_{\mathfrak g}V(x,z)\xi\rangle
+
O(||\xi||^3).
```

The Lie-algebra derivative is

```math
\langle D_{\mathfrak g}V(x,z),\eta\rangle
=
\left.
\frac{d}{d\epsilon}
\right|_{\epsilon=0}
V(\exp(\epsilon\eta)\cdot x,z).
```

Keeping first order only:

```math
V(\exp(\xi)\cdot x,z)
\approx
V(x,z) + \langle D_{\mathfrak g}V(x,z),\xi\rangle.
```

Then

```math
A_G^\pi(x,z,a)
\approx
\mathbb E_\xi[r(x,z,a,\xi)]
-
(1-\gamma)V^\pi(x,z)
+
\gamma
\left\langle
D_{\mathfrak g}V^\pi(x,z),
\mathbb E[\xi\mid x,z,a]
\right\rangle.
```

Interpretation:

```text
advantage ~= reward + alignment(value-gradient, Lie-algebra motion) - discounted baseline.
```

---

## 8. Multi-step Lie Bellman returns

One step:

```math
x_{t+1} = \exp(\xi_t)\cdot x_t.
```

Two steps:

```math
x_{t+2}
=
\exp(\xi_{t+1})\cdot x_{t+1}
=
\exp(\xi_{t+1})\exp(\xi_t)\cdot x_t.
```

`n` steps:

```math
x_{t+n}
=
\left(
 \exp(\xi_{t+n-1})
 \cdots
 \exp(\xi_{t+1})
 \exp(\xi_t)
\right)\cdot x_t.
```

For noncommutative groups,

```math
\exp(\xi_{t+1})\exp(\xi_t)
\ne
\exp(\xi_t + \xi_{t+1}).
```

Instead,

```math
\exp(\eta)\exp(\xi)
=
\exp(\operatorname{BCH}(\eta,\xi)),
```

where BCH is the Baker-Campbell-Hausdorff expansion:

```math
\operatorname{BCH}(\eta,\xi)
=
\eta + \xi
+
\frac{1}{2}[\eta,\xi]
+
\frac{1}{12}[\eta,[\eta,\xi]]
-
\frac{1}{12}[\xi,[\eta,\xi]]
+
\cdots.
```

The `n`-step Lie return is

```math
G_t^{G,n}
=
\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+
\gamma^n V_\phi(s_{t+n}^G),
```

where

```math
s_{t+n}^G
=
\left(
 \exp(\xi_{t+n-1})
 \cdots
 \exp(\xi_t)
 \cdot x_t,
 z_{t+n}
\right).
```

The `n`-step Lie advantage is

```math
\hat A_t^{G,n}
=
G_t^{G,n} - V_\phi(s_t).
```

---

## 9. Lie-Algebraic GAE

Standard GAE uses

```math
\hat A_t^{\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l},
```

with

```math
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t).
```

The Lie-geometric TD residual is

```math
\delta_t^G
=
r_t
+
\gamma V_\phi(s_{t+1}^G)
-
V_\phi(s_t),
```

where

```math
s_{t+1}^G = \left(\exp(\xi_t)\cdot x_t,z_{t+1}\right).
```

Therefore Lie-Algebraic GAE is

```math
\hat A_t^{G\text{-}\mathrm{GAE}(\gamma,\lambda)}
=
\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^G.
```

Finite-horizon form:

```math
\hat A_t^{G\text{-}\mathrm{GAE}}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\left[
 r_{t+l}
 +
 \gamma V_\phi(s_{t+l+1}^G)
 -
 V_\phi(s_{t+l}^G)
\right].
```

Recursive implementation:

```math
\hat A_t^{G\text{-}\mathrm{GAE}}
=
\delta_t^G
+
\gamma\lambda m_{t+1}
\hat A_{t+1}^{G\text{-}\mathrm{GAE}},
```

where

```math
m_{t+1}=0
```

for terminal states and

```math
m_{t+1}=1
```

otherwise.

The value target is

```math
\hat R_t^G
=
\hat A_t^{G\text{-}\mathrm{GAE}} + V_\phi(s_t).
```

---

## 10. GAE as weighted Lie n-step advantages

Define

```math
\hat A_t^{G,n}
=
\sum_{k=0}^{n-1}\gamma^k r_{t+k}
+
\gamma^n V_\phi(s_{t+n}^G)
-
V_\phi(s_t).
```

Then

```math
\hat A_t^{G\text{-}\mathrm{GAE}(\gamma,\lambda)}
=
(1-\lambda)
\sum_{n=1}^{\infty}
\lambda^{n-1}
\hat A_t^{G,n}.
```

Special cases:

```math
\lambda=0
\quad\Rightarrow\quad
\hat A_t^G = \delta_t^G.
```

```math
\lambda \to 1
\quad\Rightarrow\quad
\text{long-horizon Monte Carlo-style Lie return}.
```

---

## 11. Observed-transition version vs implicit-dynamics version

### Version A: observed-transition Lie GAE

The environment gives an observed next state `s_{t+1}`. If the geometric next state can be represented as

```math
x_{t+1}=\exp(\xi_t)\cdot x_t,
```

then

```math
\delta_t^G
=
r_t
+
\gamma V_\phi(\exp(\xi_t)\cdot x_t,z_{t+1})
-
V_\phi(x_t,z_t).
```

If the Lie representation is exact, this equals ordinary TD numerically, but the transition is represented in geometrically meaningful coordinates.

### Version B: implicit-dynamics Lie GAE

Learn

```math
\xi_\theta(s_t,a_t) \in \mathfrak g.
```

Then

```math
\hat s_{t+1}^G
=
F_\theta^G(s_t,a_t)
=
\left(
 \exp(\xi_\theta(s_t,a_t))\cdot x_t,
 \hat z_{t+1}
\right).
```

The TD residual becomes

```math
\delta_t^{G,\theta}
=
r_t
+
\gamma V_\phi(F_\theta^G(s_t,a_t))
-
V_\phi(s_t).
```

The implicit-dynamics GAE estimator is

```math
\hat A_t^{G,\theta\text{-}\mathrm{GAE}}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l}^{G,\theta}.
```

---

## 12. Training losses

### Critic TD loss

```math
\mathcal L_V(\phi,\theta)
=
\mathbb E_t
\left[
\left(
 V_\phi(s_t)
 -
 \left[
  r_t + \gamma V_{\bar\phi}(F_\theta^G(s_t,a_t))
 \right]
\right)^2
\right].
```

`V_{bar phi}` is a target network or stop-gradient copy.

### Value regression to GAE return

```math
\hat R_t^G
=
\hat A_t^{G\text{-}\mathrm{GAE}} + V_\phi(s_t).
```

Then

```math
\mathcal L_{V,\mathrm{GAE}}(\phi)
=
\mathbb E_t
\left[
\left(
 V_\phi(s_t) - \operatorname{stopgrad}(\hat R_t^G)
\right)^2
\right].
```

### Policy-gradient loss

```math
\mathcal L_\pi(\psi)
=
-
\mathbb E_t
\left[
\log\pi_\psi(a_t|s_t)
\operatorname{stopgrad}(\hat A_t^G)
\right].
```

### PPO clipped actor loss

Let

```math
\rho_t(\psi)
=
\frac{\pi_\psi(a_t|s_t)}{\pi_{\mathrm{old}}(a_t|s_t)}.
```

Then

```math
\mathcal L_\pi^{\mathrm{PPO}}(\psi)
=
-
\mathbb E_t
\left[
\min\left(
 \rho_t(\psi)\hat A_t^G,
 \operatorname{clip}(\rho_t(\psi),1-\epsilon,1+\epsilon)\hat A_t^G
\right)
\right].
```

---

## 13. Useful regularizers

A pure TD objective may not uniquely identify `xi_theta`, because multiple predicted Lie displacements can yield similar values. Add geometric regularization when possible.

### Dynamics reconstruction loss

If observed `x_{t+1}` is available:

```math
\mathcal L_{\mathrm{dyn}}
=
d_{\mathcal M}
\left(
 x_{t+1},
 \exp(\xi_\theta(s_t,a_t))\cdot x_t
\right)^2.
```

If `M = G`, use the group geodesic-style distance:

```math
d_G(X,Y)
=
\left\|\log(X^{-1}Y)\right\|.
```

### Lie-Taylor consistency loss

For small `epsilon`:

```math
\mathcal L_{\mathrm{Taylor}}
=
\left[
 V_\phi(\exp(\epsilon\xi)\cdot x,z)
 -
 V_\phi(x,z)
 -
 \epsilon\langle D_{\mathfrak g}V_\phi(x,z),\xi\rangle
\right]^2.
```

### Symmetry regularization

If the task has a true symmetry subgroup `H subset G`, impose value invariance:

```math
V_\phi(h\cdot x,z)=V_\phi(x,z).
```

For dynamics equivariance:

```math
\xi_\theta(h\cdot s,h\cdot a)
=
\operatorname{Ad}_h \xi_\theta(s,a).
```

The adjoint appears because

```math
h\exp(\xi)h^{-1}
=
\exp(\operatorname{Ad}_h\xi).
```

Important distinction:

```text
Lie group as dynamics geometry is not the same thing as Lie group as task symmetry.
```

You can use the group to model transitions even when the value is not invariant under the group.

---

## 14. Continuous-time limit: Lie-HJB equation

Let timestep be `Delta t`, discount be

```math
\gamma = e^{-\rho\Delta t},
```

and reward be

```math
r_t = \Delta t\,\ell(s_t,a_t),
```

where `ell` is reward rate.

Let

```math
\xi_t = \Delta t\,u(s_t,a_t),
```

where

```math
u(s,a) \in \mathfrak g
```

is the instantaneous Lie-algebra velocity.

Then

```math
x_{t+\Delta t}
=
\exp(\Delta t\,u(s_t,a_t))\cdot x_t.
```

The continuous-time policy equation is

```math
\rho V^\pi(s)
=
\mathbb E_{a\sim\pi(\cdot|s)}
\left[
 \ell(s,a)
 +
 \mathcal L_{u(s,a)}V^\pi(s)
 +
 D_zV^\pi(s)[\dot z(s,a)]
\right].
```

If there is no residual variable `z`, this simplifies to

```math
\rho V^\pi(x)
=
\mathbb E_{a\sim\pi(\cdot|x)}
\left[
 \ell(x,a)
 +
 \mathcal L_{u(x,a)}V^\pi(x)
\right].
```

The optimal control form is

```math
\rho V^*(x)
=
\sup_a
\left[
 \ell(x,a)
 +
 \mathcal L_{u(x,a)}V^*(x)
\right].
```

The instantaneous Lie advantage is

```math
\mathcal A_G^\pi(x,a)
=
\ell(x,a)
+
\mathcal L_{u(x,a)}V^\pi(x)
-
\rho V^\pi(x).
```

Under the policy,

```math
\mathbb E_{a\sim\pi(\cdot|x)}[\mathcal A_G^\pi(x,a)] = 0.
```

---

## 15. Suggested Python module layout

```text
lie_bellman_rl/
  __init__.py
  groups/
    __init__.py
    base.py              # LieGroup interface
    so2.py               # SO(2) implementation
    se2.py               # SE(2) implementation
    so3.py               # optional
    se3.py               # optional
  models/
    __init__.py
    policy.py            # pi_psi(a | s)
    value.py             # V_phi(s)
    lie_dynamics.py      # xi_theta(s,a), optional z model
  algos/
    __init__.py
    gae.py               # Lie-Algebraic GAE
    losses.py            # TD, PPO, regularizers
    rollout.py           # rollout buffer
  examples/
    ppo_lie_bellman.py
  tests/
    test_so2.py
    test_se2.py
    test_gae.py
```

---

## 16. Core interfaces

### LieGroup base class

```python
from abc import ABC, abstractmethod
import torch

class LieGroup(ABC):
    @property
    @abstractmethod
    def algebra_dim(self) -> int:
        pass

    @abstractmethod
    def exp(self, xi: torch.Tensor) -> torch.Tensor:
        """Map algebra element xi to group element g."""
        pass

    @abstractmethod
    def log(self, g: torch.Tensor) -> torch.Tensor:
        """Map group element g to algebra element xi."""
        pass

    @abstractmethod
    def act(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Apply group element g to geometric state x."""
        pass

    @abstractmethod
    def compose(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """Group composition g1 * g2."""
        pass

    @abstractmethod
    def inverse(self, g: torch.Tensor) -> torch.Tensor:
        """Group inverse."""
        pass

    def distance(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Optional manifold/group distance."""
        raise NotImplementedError

    def adjoint(self, g: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """Optional adjoint action Ad_g xi."""
        raise NotImplementedError
```

### Lie dynamics model

```python
import torch
import torch.nn as nn

class LieDynamics(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, algebra_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, algebra_dim),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))
```

### Lie transition function

```python
def lie_transition(group, x, z_next, xi):
    g = group.exp(xi)
    x_next = group.act(g, x)
    return x_next, z_next
```

For a learned dynamics model:

```python
def learned_lie_transition(group, dynamics, state, x, z_next_hat, action):
    xi = dynamics(state, action)
    g = group.exp(xi)
    x_next = group.act(g, x)
    return x_next, z_next_hat, xi
```

---

## 17. Lie-Algebraic TD residual implementation

Observed-transition version:

```python
def lie_td_residual_observed(value_fn, group, x, z, xi, z_next, reward, done, gamma):
    state = pack_state(x, z)
    g = group.exp(xi)
    x_next_g = group.act(g, x)
    next_state_g = pack_state(x_next_g, z_next)

    v = value_fn(state).squeeze(-1)
    with torch.no_grad():
        v_next = value_fn(next_state_g).squeeze(-1)
        target = reward + gamma * (1.0 - done.float()) * v_next
    delta = target - v
    return delta, target
```

Implicit-dynamics version:

```python
def lie_td_residual_learned(value_fn, target_value_fn, dynamics, group, state, x, z, action, reward, done, gamma, z_next_hat):
    xi = dynamics(state, action)
    g = group.exp(xi)
    x_next_g = group.act(g, x)
    next_state_g = pack_state(x_next_g, z_next_hat)

    v = value_fn(state).squeeze(-1)
    with torch.no_grad():
        v_next = target_value_fn(next_state_g).squeeze(-1)
        target = reward + gamma * (1.0 - done.float()) * v_next
    delta = target - v
    return delta, target, xi, next_state_g
```

---

## 18. Lie-Algebraic GAE implementation

Input shapes:

```text
rewards: [T, B]
values: [T, B]
next_values: [T, B]
dones: [T, B]
gamma: scalar
lam: scalar
```

Where

```math
\delta_t^G = r_t + \gamma(1-d_t)V(s_{t+1}^G) - V(s_t).
```

Implementation:

```python
import torch

def compute_lie_gae(rewards, values, next_values_g, dones, gamma: float, lam: float):
    """
    Compute Lie-Algebraic GAE.

    Args:
        rewards: Tensor [T, B]
        values: Tensor [T, B], V(s_t)
        next_values_g: Tensor [T, B], V(s_{t+1}^G)
        dones: Tensor [T, B], 1 if terminal else 0
        gamma: discount factor
        lam: GAE lambda

    Returns:
        advantages: Tensor [T, B]
        returns: Tensor [T, B]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(rewards[0])

    for t in reversed(range(T)):
        nonterminal = 1.0 - dones[t].float()
        delta = rewards[t] + gamma * nonterminal * next_values_g[t] - values[t]
        last_adv = delta + gamma * lam * nonterminal * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns
```

---

## 19. Minimal PPO-style training loop

```python
for iteration in range(num_iterations):
    # 1. Collect rollout with current policy.
    rollout = collect_rollout(env, policy, value_fn, dynamics, group)

    # 2. Build Lie-geometric next states.
    # Observed version: infer xi from x_t and x_{t+1} if possible.
    # Learned version: xi = dynamics(s_t, a_t), x_next_g = group.act(group.exp(xi), x_t).
    next_states_g = build_lie_next_states(rollout, group, dynamics)

    # 3. Evaluate values.
    values = value_fn(rollout.states).squeeze(-1)
    with torch.no_grad():
        next_values_g = value_fn(next_states_g).squeeze(-1)

    # 4. Compute Lie-Algebraic GAE.
    advantages, returns = compute_lie_gae(
        rewards=rollout.rewards,
        values=values,
        next_values_g=next_values_g,
        dones=rollout.dones,
        gamma=gamma,
        lam=gae_lambda,
    )

    # 5. Normalize advantages.
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # 6. PPO minibatch updates.
    for epoch in range(ppo_epochs):
        for batch in rollout.minibatches(advantages, returns):
            logprob = policy.log_prob(batch.states, batch.actions)
            ratio = torch.exp(logprob - batch.old_logprobs)

            unclipped = ratio * batch.advantages
            clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_pred = value_fn(batch.states).squeeze(-1)
            value_loss = ((value_pred - batch.returns) ** 2).mean()

            entropy_loss = -policy.entropy(batch.states).mean()

            loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_grad_norm)
            optimizer.step()
```

---

## 20. Example: SO(2)

`SO(2)` is the simplest possible implementation.

Lie algebra:

```math
\mathfrak{so}(2) \cong \mathbb R.
```

Group element:

```math
g = R(\theta).
```

Exponential:

```math
\exp(\theta)
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}.
```

Action on a 2D vector:

```math
x' = R(\theta)x.
```

Implementation sketch:

```python
import torch

class SO2:
    @property
    def algebra_dim(self):
        return 1

    def exp(self, xi):
        theta = xi.squeeze(-1)
        c = torch.cos(theta)
        s = torch.sin(theta)
        row1 = torch.stack([c, -s], dim=-1)
        row2 = torch.stack([s, c], dim=-1)
        return torch.stack([row1, row2], dim=-2)

    def log(self, g):
        theta = torch.atan2(g[..., 1, 0], g[..., 0, 0])
        return theta.unsqueeze(-1)

    def act(self, g, x):
        return torch.matmul(g, x.unsqueeze(-1)).squeeze(-1)

    def compose(self, g1, g2):
        return torch.matmul(g1, g2)

    def inverse(self, g):
        return g.transpose(-1, -2)
```

---

## 21. Example: SE(2)

`SE(2)` is useful for planar robot pose.

A pose is

```math
T =
\begin{bmatrix}
R & t \\
0 & 1
\end{bmatrix}
\in SE(2).
```

A Lie algebra vector can be represented as

```math
\xi = (v_x,v_y,\omega) \in \mathbb R^3.
```

The transition is

```math
T_{t+1} = \exp(\xi_t)T_t
```

for left action, or

```math
T_{t+1} = T_t\exp(\xi_t)
```

for right action. Choose one convention and keep it consistent.

For many RL settings, using right action can be convenient because the action is expressed in the robot/body frame. Left action can be convenient if the displacement is expressed in the world frame.

---

## 22. Practical design choices

### Choice 1: what is `x`?

Examples:

```text
SO(2): heading angle or 2D orientation vector
SO(3): object orientation
SE(2): planar robot pose
SE(3): 6D rigid-body pose
R^n: ordinary translation group
SE(3) x R^k: pose plus Euclidean variables
```

### Choice 2: what is `z`?

Examples:

```text
joint velocities
contact flags
object attributes
sensor readings
latent memory state
goal vector
previous action
```

### Choice 3: observed or learned Lie displacement?

Observed version:

```text
Use environment next state and compute or fit xi_t.
Good for debugging and validating the method.
```

Learned version:

```text
Use xi_theta(s,a) inside the Bellman target.
This is the more novel implicit-dynamics version.
```

### Choice 4: stop gradients through the target?

For stable critic learning, usually use a target value network or stop-gradient target:

```math
r_t + \gamma V_{\bar\phi}(F_\theta^G(s_t,a_t)).
```

If training `xi_theta` through the value target, be careful. Otherwise `xi_theta` can learn adversarial transitions that artificially reduce TD error without matching true dynamics.

Recommended early-stage objective:

```math
\mathcal L
=
\mathcal L_{V,\mathrm{GAE}}
+
\alpha \mathcal L_{\mathrm{dyn}}
+
\beta \mathcal L_{\mathrm{Taylor}}
+
\eta \mathcal L_{\pi}.
```

---

## 23. Recommended MVP

The fastest path to a working prototype:

1. Implement `SO2` or `SE2` first.
2. Use a simple environment where orientation or pose matters.
3. Decompose state into `(x,z)`.
4. Use observed transitions first.
5. Implement `delta_t^G` and `compute_lie_gae`.
6. Plug Lie-GAE into PPO.
7. Confirm it matches standard PPO when the Lie transition exactly reconstructs the observed next state.
8. Add learned `xi_theta(s,a)`.
9. Add `L_dyn` to prevent degenerate implicit transitions.
10. Compare standard PPO vs Lie-GAE PPO vs learned Lie-GAE PPO.

---

## 24. Invariants and tests

### SO(2) tests

```python
def test_so2_exp_is_rotation(group):
    xi = torch.randn(128, 1)
    g = group.exp(xi)
    eye = torch.eye(2, device=g.device).expand(128, 2, 2)
    assert torch.allclose(g.transpose(-1, -2) @ g, eye, atol=1e-5)


def test_so2_log_exp_inverse(group):
    xi = 0.1 * torch.randn(128, 1)
    xi_rec = group.log(group.exp(xi))
    assert torch.allclose(xi, xi_rec, atol=1e-5)
```

### GAE tests

```python
def test_lie_gae_zero_rewards_zero_values():
    T, B = 8, 4
    rewards = torch.zeros(T, B)
    values = torch.zeros(T, B)
    next_values = torch.zeros(T, B)
    dones = torch.zeros(T, B)
    adv, ret = compute_lie_gae(rewards, values, next_values, dones, 0.99, 0.95)
    assert torch.allclose(adv, torch.zeros_like(adv))
    assert torch.allclose(ret, torch.zeros_like(ret))
```

### Equivalence test

If `s_{t+1}^G == s_{t+1}` exactly, Lie-GAE should equal standard GAE.

```python
def test_lie_gae_equals_standard_gae_when_next_states_match():
    # Build next_values_g from actual observed next states.
    # Compare compute_lie_gae(...) to ordinary compute_gae(...).
    pass
```

---

## 25. Main equations to keep near the code

Lie transition:

```math
s_{t+1}^G
=
F_\theta^G(s_t,a_t)
=
\left(\exp(\xi_\theta(s_t,a_t))\cdot x_t,z_{t+1}\right).
```

Lie-Algebraic Bellman equation:

```math
V^\pi(s)
=
\mathbb E_{a\sim\pi}
\mathbb E_{\xi,z'}
\left[
 r(s,a,\xi,z')
 +
 \gamma V^\pi(F^G(s,a,\xi,z'))
\right].
```

Lie-Algebraic advantage:

```math
A_G^\pi(s,a)
=
\mathbb E_{\xi,z'}
\left[
 r(s,a,\xi,z')
 +
 \gamma V^\pi(F^G(s,a,\xi,z'))
 -
 V^\pi(s)
\right].
```

Lie-Algebraic TD residual:

```math
\delta_t^G
=
r_t
+
\gamma V_\phi(F_\theta^G(s_t,a_t))
-
V_\phi(s_t).
```

Lie-Algebraic GAE:

```math
\hat A_t^{G\text{-}\mathrm{GAE}}
=
\sum_{l=0}^{T-t-1}
(\gamma\lambda)^l
\delta_{t+l}^G.
```

Continuous-time instantaneous advantage:

```math
\mathcal A_G^\pi(s,a)
=
\ell(s,a)
+
\mathcal L_{u(s,a)}V^\pi(s)
-
\rho V^\pi(s).
```

---

## 26. Codex starter prompt

Use the following prompt to ask Codex to start implementing the project:

```text
Implement a PyTorch package for Lie-Algebraic Bellman backups and Lie-Algebraic GAE.

Use the following module layout:

lie_bellman_rl/
  groups/base.py
  groups/so2.py
  groups/se2.py
  models/value.py
  models/policy.py
  models/lie_dynamics.py
  algos/gae.py
  algos/losses.py
  tests/test_so2.py
  tests/test_gae.py

Core requirements:
1. Implement a LieGroup base interface with exp, log, act, compose, inverse, distance, and adjoint hooks.
2. Implement SO2 first with batched PyTorch tensors.
3. Implement compute_lie_gae(rewards, values, next_values_g, dones, gamma, lam).
4. Implement LieDynamics as an MLP that predicts xi_theta(s,a).
5. Implement functions to build Lie next states using group.exp(xi) and group.act(g,x).
6. Implement losses for value regression, TD residual, PPO clipped actor loss, dynamics reconstruction loss, and optional Taylor consistency loss.
7. Add tests showing that SO2 exp produces orthogonal matrices, log(exp(xi)) approximately equals xi for small xi, and Lie-GAE equals standard GAE when Lie next states match observed next states.
8. Keep all functions batched, differentiable, and device/dtype preserving.
```
