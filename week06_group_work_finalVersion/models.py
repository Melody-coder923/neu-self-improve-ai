"""
Compact MLP ActorCritic (paper Appendix A: "compact MLP").
hidden_dim=64, orthogonal init.

Two variants:
- ActorCritic: shared backbone (for PPO, REINFORCE, pretrain)
- SeparateActorCritic: separate actor/critic (for Poly-PPO, Table 3 has
  different actor lr=1e-5 and critic lr=1e-4)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class ActorCritic(nn.Module):
    """Shared-backbone version for PPO, REINFORCE, pretrain."""

    def __init__(self, obs_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
        )
        self.actor_head = layer_init(
            nn.Linear(hidden_dim, num_actions), std=0.01)
        self.critic_head = layer_init(
            nn.Linear(hidden_dim, 1), std=1.0)

    def forward(self, x):
        h = self.shared(x)
        return self.actor_head(h), self.critic_head(h)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, x):
        _, v = self.forward(x)
        return v.squeeze(-1)


class SeparateActorCritic(nn.Module):
    """Separate actor/critic for Poly-PPO (Table 3: actor lr != critic lr)."""

    def __init__(self, obs_dim, num_actions, hidden_dim=64):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, num_actions), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, hidden_dim)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_dim, 1), std=1.0),
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze(-1)

    def get_value(self, x):
        return self.critic(x).squeeze(-1)

    def actor_parameters(self):
        return self.actor.parameters()

    def critic_parameters(self):
        return self.critic.parameters()
