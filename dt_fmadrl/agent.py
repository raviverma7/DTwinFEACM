from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(z, -40, 40))
    return exp / exp.sum(axis=-1, keepdims=True)


class LinearActorCritic:
    """Small auditable actor-critic used by every federated client."""

    def __init__(self, state_dim: int, rng: np.random.Generator):
        self.actor = rng.normal(0, 0.02, size=(state_dim + 1, 7))
        self.critic = np.zeros(state_dim + 1)

    @staticmethod
    def features(states: np.ndarray) -> np.ndarray:
        return np.column_stack((states, np.ones(len(states))))

    def probabilities(self, states: np.ndarray, feasible=None) -> np.ndarray:
        logits = self.features(states) @ self.actor
        if feasible is not None:
            logits = np.where(feasible, logits, -1e30)
        probs = softmax(logits) * (feasible if feasible is not None else 1)
        return probs / probs.sum(axis=1, keepdims=True)

    def actions(self, states: np.ndarray, rng: np.random.Generator, explore: bool = True) -> np.ndarray:
        probs = self.probabilities(states)
        if not explore:
            return np.argmax(probs, axis=1)
        return np.array([rng.choice(7, p=p) for p in probs], dtype=int)

    def update(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray,
               next_states: np.ndarray, gamma: float, actor_lr: float,
               critic_lr: float, entropy: float, terminated=False, feasible=None) -> None:
        x, nx = self.features(states), self.features(next_states)
        values, next_values = x @ self.critic, nx @ self.critic
        td = np.clip(rewards + gamma * (1-np.asarray(terminated,dtype=float)) * next_values - values, -10.0, 10.0)
        self.critic += critic_lr * np.mean(td[:, None] * x, axis=0)
        self.critic = np.clip(np.nan_to_num(self.critic), -100.0, 100.0)
        probs = self.probabilities(states, feasible)
        chosen = np.eye(7)[actions]
        policy_gradient = x.T @ (td[:, None] * (chosen - probs)) / len(states)
        logp = np.log(np.maximum(probs, 1e-30))
        h = -np.sum(probs*logp, axis=1, keepdims=True)
        policy_gradient += entropy * x.T @ (-probs*(logp+h)) / len(states)
        self.actor += actor_lr * np.clip(policy_gradient, -5.0, 5.0)
        self.actor = np.clip(np.nan_to_num(self.actor), -100.0, 100.0)

    def parameters(self) -> tuple[np.ndarray, np.ndarray]:
        return self.actor.copy(), self.critic.copy()

    def set_parameters(self, params: tuple[np.ndarray, np.ndarray]) -> None:
        self.actor[...] = params[0]
        self.critic[...] = params[1]
