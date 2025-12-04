import torch
import numpy as np
import gymnasium as gym

def collect_random_data(env, steps=5000):
    states = []
    actions = []
    rewards = []
    next_states = []

    s, _ = env.reset()
    for _ in range(steps):
        a = env.action_space.sample()

        s_new, r, done, _, _ = env.step(a)

        states.append(s)
        actions.append(a)
        rewards.append(r)
        next_states.append(s_new)

        if done:
            s, _ = env.reset()
        else:
            s = s_new
    states = np.array(states)
    actions = np.array(actions).reshape(-1, 1)
    inputs = np.concatenate([states, actions], axis=1)
    next_states = np.array(next_states)
    rewards = np.array(rewards).reshape(-1, 1)
    # convert to tensors
    inputs = torch.tensor(inputs, dtype=torch.float32)
    targets = torch.tensor(np.concatenate([next_states, rewards], axis=1), dtype=torch.float32)

    return inputs, targets

env = gym.make("CartPole-v1")
inputs, targets = collect_random_data(env, steps = 100000)
np.savez('data.npz', inputs=inputs, targets=targets)
