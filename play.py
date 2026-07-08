"""
play.py  –  Watch your trained DQN agent play Flappy Bird
==========================================================
Run:
    python play.py                         # uses best_model.pt by default
    python play.py --model checkpoints/model_ep400.pt
    python play.py --episodes 5
"""

import argparse
import numpy as np
import gymnasium
import flappy_bird_gymnasium              # registers the env

from dqn_agent import DQNAgent

DEFAULT_MODEL = "checkpoints/best_model.pt"
USE_LIDAR     = False                     # must match what you trained with


def parse_args():
    parser = argparse.ArgumentParser(description="Watch DQN agent play Flappy Bird")
    parser.add_argument("--model",    type=str, default=DEFAULT_MODEL,
                        help="Path to saved model (.pt)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes to play")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable visual rendering (for headless eval)")
    return parser.parse_args()


def main():
    args = parse_args()

    render_mode = "rgb_array" if args.no_render else "human"
    env = gymnasium.make("FlappyBird-v0", render_mode=render_mode, use_lidar=USE_LIDAR)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n

    # Load agent  (epsilon=0 → purely greedy)
    agent = DQNAgent(
        state_dim         = obs_dim,
        action_dim        = act_dim,
        epsilon_start     = 0.0,
        epsilon_end       = 0.0,
        epsilon_decay     = 1.0,
    )
    agent.load(args.model)
    agent.epsilon = 0.0      # force greedy play

    print(f"\n▶  Playing {args.episodes} episode(s) with model: {args.model}\n")

    scores = []
    for ep in range(1, args.episodes + 1):
        obs, _       = env.reset()
        total_reward = 0.0
        steps        = 0

        while True:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps        += 1
            if terminated or truncated:
                break

        score = info.get("score", "?")
        scores.append(total_reward)
        print(f"  Episode {ep:>3}  |  Reward: {total_reward:>8.2f}  |  Pipes passed: {score}  |  Steps: {steps}")

    env.close()

    print(f"\n── Summary ──────────────────────────")
    print(f"  Episodes     : {args.episodes}")
    print(f"  Avg Reward   : {np.mean(scores):.2f}")
    print(f"  Best Reward  : {np.max(scores):.2f}")
    print(f"─────────────────────────────────────\n")


if __name__ == "__main__":
    main()
