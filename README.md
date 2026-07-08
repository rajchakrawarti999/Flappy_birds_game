# 🐦 Flappy Bird — Deep Q-Network (DQN) AI

An AI agent that learns to play Flappy Bird using Deep Reinforcement Learning (DQN).  
Trained from scratch using the `flappy-bird-gymnasium` environment.

---

## 🎮 Demo

> Agent playing after 2000 episodes of training!

*(Add a GIF or screenshot here)*

---

## 🧠 How It Works

This project uses **Deep Q-Network (DQN)** — a Reinforcement Learning algorithm where:

- The **agent** (bird) observes the game state
- It takes an **action** (flap or do nothing)
- It receives a **reward** based on survival and pipe passing
- Over time, it learns the **best strategy** to maximize reward

### Rewards
| Event | Reward |
|-------|--------|
| Every frame alive | +0.1 |
| Pass a pipe | +1.0 |
| Die | -1.0 |
| Hit top of screen | -0.5 |

---

## 🏗️ Project Structure

```
Flappy_Birds_game/
│
├── dqn_agent.py      # DQN Agent — Neural Network, Replay Buffer, Target Network
├── train.py          # Training loop
├── play.py           # Watch trained agent play
├── checkpoints/      # Saved models (auto-created during training)
│   └── best_model.pt
└── README.md
```

---

## ⚙️ Installation

> Python 3.11 recommended

```bash
pip install pygame flappy-bird-gymnasium gymnasium torch
```

---

## 🚀 Usage

### Train the agent
```bash
python train.py
```

### Watch the agent play
```bash
python play.py
```

### Watch a specific checkpoint
```bash
python play.py --model checkpoints/model_ep400.pt
```

---

## 🧬 DQN Architecture

```
Input Layer  →  Linear(128)  →  ReLU
             →  Linear(128)  →  ReLU
             →  Linear(64)   →  ReLU
             →  Output(2)
```

**Input:** 12 game state values (pipe positions, bird velocity, rotation)  
**Output:** 2 actions — Flap / Do Nothing

### Key Features
- ✅ Experience Replay Buffer (50,000 transitions)
- ✅ Target Network (synced every 500 steps)
- ✅ Epsilon-Greedy Exploration (1.0 → 0.01)
- ✅ Gradient Clipping for stable training
- ✅ Auto GPU/CPU detection

---

## 📊 Training Results

| Episodes | Avg Reward | Pipes Passed |
|----------|-----------|--------------|
| 0–200    | Negative  | 0–1          |
| 500–800  | +2 to +5  | 1–5          |
| 1000+    | +10 to +20| 5–20         |
| 2000     | +29.51    | 20+          |

---

## 🛠️ Tech Stack

- **Python 3.11**
- **PyTorch** — Neural Network
- **Gymnasium** — RL Environment
- **Pygame** — Game Rendering
- **flappy-bird-gymnasium** — Flappy Bird Environment

---

## 👤 Author

**Raj Chakrawarti**  
GitHub: [@rajchakrawarti999](https://github.com/rajchakrawarti999)
