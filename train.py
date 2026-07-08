"""
train.py  –  DQN Training for Flappy Bird
==========================================
Run:
    python train.py

Requirements:
    pip install flappy_bird_gymnasium
    pip install pyyaml
    pip install torch gymnasium
"""

import os
import time
import numpy as np
import gymnasium
import flappy_bird_gymnasium          # registers the env

from dqn_agent import DQNAgent

# ─────────────────────────────────────────
#  Hyper-parameters
# ─────────────────────────────────────────
CONFIG = {
    # Environment
    "use_lidar":        False,         # True = 180-dim LIDAR obs, False = 12-dim game state
    # Training
    "num_episodes":     2000,
    "max_steps":        10_000,        # safety cap per episode
    "lr":               1e-3,
    "gamma":            0.99,
    "epsilon_start":    1.0,
    "epsilon_end":      0.01,
    "epsilon_decay":    0.995,
    "batch_size":       64,
    "buffer_capacity":  50_000,
    "target_update_freq": 500,         # steps between target-net syncs
    # Saving
    "save_dir":         "checkpoints",
    "save_every":       200,           # save model every N episodes
    "log_every":        10,            # print stats every N episodes
}

BEST_MODEL_PATH = os.path.join(CONFIG["save_dir"], "best_model.pt")
LAST_MODEL_PATH = os.path.join(CONFIG["save_dir"], "last_model.pt")


def make_env(render: bool = False) -> gymnasium.Env:
    mode = "human" if render else "rgb_array"
    return gymnasium.make(
        "FlappyBird-v0",
        render_mode=mode,
        use_lidar=CONFIG["use_lidar"],
    )


def main():
    os.makedirs(CONFIG["save_dir"], exist_ok=True)

    # ── Setup ───────────────────────────
    env       = make_env(render=False)
    obs_dim   = env.observation_space.shape[0]
    act_dim   = env.action_space.n

    print(f"\n{'='*50}")
    print(f"  Flappy Bird DQN Training")
    print(f"  Observation dim : {obs_dim}")
    print(f"  Action dim      : {act_dim}")
    print(f"  Device          : auto (GPU if available)")
    print(f"{'='*50}\n")

    agent = DQNAgent(
        state_dim         = obs_dim,
        action_dim        = act_dim,
        lr                = CONFIG["lr"],
        gamma             = CONFIG["gamma"],
        epsilon_start     = CONFIG["epsilon_start"],
        epsilon_end       = CONFIG["epsilon_end"],
        epsilon_decay     = CONFIG["epsilon_decay"],
        batch_size        = CONFIG["batch_size"],
        target_update_freq= CONFIG["target_update_freq"],
        buffer_capacity   = CONFIG["buffer_capacity"],
    )

    # ── Training Loop ───────────────────
    episode_rewards = []
    best_avg_reward = -float("inf")
    start_time      = time.time()

    for ep in range(1, CONFIG["num_episodes"] + 1):

        obs, _    = env.reset()
        total_reward = 0.0
        total_loss   = 0.0
        loss_count   = 0

        for step in range(CONFIG["max_steps"]):
            action     = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = terminated or truncated

            agent.store(obs, action, reward, next_obs, float(done))
            loss = agent.learn()
            if loss is not None:
                total_loss += loss
                loss_count += 1

            obs          = next_obs
            total_reward += reward

            if done:
                break

        episode_rewards.append(total_reward)

        # ── Logging ─────────────────────
        if ep % CONFIG["log_every"] == 0:
            avg_reward = np.mean(episode_rewards[-50:])   # last-50 avg
            avg_loss   = total_loss / max(loss_count, 1)
            elapsed    = time.time() - start_time
            print(
                f"Ep {ep:>5}/{CONFIG['num_episodes']}  |  "
                f"Reward: {total_reward:>7.2f}  |  "
                f"Avg(50): {avg_reward:>7.2f}  |  "
                f"Loss: {avg_loss:.4f}  |  "
                f"ε: {agent.epsilon:.3f}  |  "
                f"Time: {elapsed/60:.1f}m"
            )

            # Save best model
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(BEST_MODEL_PATH)
                print(f"  ★  New best avg reward: {best_avg_reward:.2f}")

        # ── Periodic checkpoint ─────────
        if ep % CONFIG["save_every"] == 0:
            ckpt = os.path.join(CONFIG["save_dir"], f"model_ep{ep}.pt")
            agent.save(ckpt)

    # ── Final save ──────────────────────
    agent.save(LAST_MODEL_PATH)
    env.close()

    print("\n" + "="*50)
    print(f"  Training complete!")
    print(f"  Best Avg Reward : {best_avg_reward:.2f}")
    print(f"  Model saved at  : {BEST_MODEL_PATH}")
    print("="*50)


if __name__ == "__main__":
    main()



# py -3.11 play.py