"""
REINFORCE (Monte Carlo Policy Gradient) on CartPole-v1
using BareTensor autograd engine.

Proves the framework works for RL — the computation graph
is built from environment trajectories, not labeled data.
"""
import time
import numpy as np
import gymnasium as gym
from baretensor import Tensor, Linear, Module, Adam


class PolicyNetwork(Module):
    """Simple 2-layer policy: state -> action probabilities."""

    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.fc1 = Linear(state_dim, hidden_dim)
        self.fc2 = Linear(hidden_dim, action_dim)


    def forward(self, x):
        x = self.fc1(x).relu()
        logits = self.fc2(x)
        probs = logits.softmax(axis=-1)
        return probs

    def __call__(self, x):
        return self.forward(x)


def compute_discounted_returns(rewards, gamma=0.99):
    """Compute discounted returns G_t = sum(gamma^k * r_{t+k}) and normalize."""
    returns = np.zeros(len(rewards), dtype=np.float32)
    G = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    # Normalize returns for stable training
    std = returns.std()
    if std > 1e-8:
        returns = (returns - returns.mean()) / std
    return returns


def train():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]    # 4
    action_dim = env.action_space.n                 # 2
    hidden_dim = 128

    policy = PolicyNetwork(state_dim, hidden_dim, action_dim)
    optimizer = Adam(policy.parameters(), lr=0.01)

    total_params = sum(p.data.size for p in policy.parameters())
    print(f"Policy Network: {state_dim} -> {hidden_dim} -> {action_dim}")
    print(f"Total parameters: {total_params:,}")
    print(f"\nTraining REINFORCE on CartPole-v1...")
    print(f"{'Episode':>8} {'Length':>8} {'Total Reward':>12} {'Avg Loss':>10}")
    print("-" * 50)

    num_episodes = 500
    recent_lengths = []

    t0 = time.time()

    for episode in range(num_episodes):
        states, actions, rewards = [], [], []
        state, _ = env.reset(seed=episode)
        done = False

        # Collect one episode
        while not done:
            state_t = Tensor(state.astype(np.float32).reshape(1, -1))
            probs = policy(state_t)
            probs_data = probs.data[0]

            # Sample action from policy distribution
            action = np.random.choice(action_dim, p=probs_data)
            next_state, reward, terminated, truncated, _ = env.step(int(action))
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state

        # Compute discounted returns
        returns = compute_discounted_returns(rewards)

        # REINFORCE: accumulate gradients over the episode
        # Use the autograd engine to propagate through the network
        # by setting probs.grad and calling backward()
        for t in range(len(states)):
            state_t = Tensor(states[t].astype(np.float32).reshape(1, -1))
            probs = policy(state_t)

            # Gradient of -log(prob_action) * return w.r.t. probs
            prob_a = probs.data[0, actions[t]] + 1e-8
            grad = np.zeros_like(probs.data)
            grad[0, actions[t]] = -returns[t] / prob_a

            # Set the gradient and propagate through the computation graph
            probs.grad = grad
            # Dummy loss tensor so backward() does topological traversal
            dummy = Tensor(np.array(0.0), parents=(probs,), requires_grad=True)
            dummy.backward()

        optimizer.step()
        optimizer.zero_grad()

        avg_loss = float(np.mean([abs(r) for r in returns]))
        episode_return = sum(rewards)
        recent_lengths.append(len(states))

        if (episode + 1) % 50 == 0:
            avg_len = np.mean(recent_lengths[-50:])
            print(f"{episode+1:8d} {len(states):8d} {episode_return:12.2f} {avg_loss:10.4f}")

    env.close()
    elapsed = time.time() - t0

    # Final evaluation
    print("\n" + "=" * 50)
    print(f"Training complete in {elapsed:.1f}s.")
    print(f"Evaluating...")

    eval_env = gym.make('CartPole-v1')
    eval_episodes = 20
    eval_lengths = []
    for i in range(eval_episodes):
        state, _ = eval_env.reset(seed=1000 + i)
        done = False
        length = 0
        while not done:
            state_t = Tensor(state.astype(np.float32).reshape(1, -1))
            probs = policy(state_t)
            action = np.argmax(probs.data[0])
            next_state, _, terminated, truncated, _ = eval_env.step(int(action))
            done = terminated or truncated
            state = next_state
            length += 1
        eval_lengths.append(length)

    eval_env.close()

    avg_eval = np.mean(eval_lengths)
    max_eval = max(eval_lengths)
    print(f"Evaluation ({eval_episodes} episodes):")
    print(f"  Average episode length: {avg_eval:.1f}")
    print(f"  Max episode length:     {max_eval}")
    print(f"  Solved (>=195 avg):     {'YES' if avg_eval >= 195 else 'NO'}")

    return avg_eval


if __name__ == "__main__":
    train()
