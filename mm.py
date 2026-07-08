# =============================================================================
# House Price Prediction using Artificial Neural Network (ANN)
# Dataset: Boston Housing (House_Dataset.csv)
# Target: MEDV (Median value of owner-occupied homes in $1000s)
# =============================================================================

# ── 1. Imports ────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ── 2. Load & Explore Data ────────────────────────────────────────────────────
df = pd.read_csv("House_Dataset.csv")

print("── Dataset Overview ──")
print(f"Shape        : {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nColumns: {list(df.columns)}")

# ── 3. Exploratory Data Analysis (EDA) ───────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Scatter: number of rooms vs. house price
axes[0].set_title("RM vs MEDV", fontsize=13)
sns.scatterplot(x=df["RM"], y=df["MEDV"], ax=axes[0], alpha=0.6)
axes[0].set_xlabel("Average Number of Rooms (RM)")
axes[0].set_ylabel("Median House Price (MEDV)")

# Correlation heatmap
corr_cols = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
             'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
axes[1].set_title("Feature Correlation Heatmap", fontsize=13)
sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
            cmap="coolwarm", ax=axes[1])

plt.tight_layout()
plt.savefig("eda_plots.png", dpi=150)
plt.show()

# ── 4. Feature Selection & Split ─────────────────────────────────────────────
# Features chosen based on highest correlation with MEDV
FEATURES = ['RM', 'LSTAT', 'PTRATIO', 'TAX', 'INDUS']
TARGET   = 'MEDV'

X = df[FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTrain size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

# ── 5. Feature Scaling ────────────────────────────────────────────────────────
# Fit scaler on training data only to prevent data leakage
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Convert to PyTorch Tensors ─────────────────────────────────────────────
X_train_tensor = torch.tensor(X_train_scaled,    dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values,    dtype=torch.float32).view(-1, 1)
X_test_tensor  = torch.tensor(X_test_scaled,     dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test.values,     dtype=torch.float32).view(-1, 1)

# ── 7. DataLoaders ────────────────────────────────────────────────────────────
BATCH_SIZE = 32

train_loader = DataLoader(
    TensorDataset(X_train_tensor, y_train_tensor),
    batch_size=BATCH_SIZE, shuffle=True
)
test_loader = DataLoader(
    TensorDataset(X_test_tensor, y_test_tensor),
    batch_size=BATCH_SIZE
)

# ── 8. Define the ANN Model ───────────────────────────────────────────────────
class HousePriceANN(nn.Module):
    """
    Feed-forward ANN for house price regression.

    Architecture:
        Input(5) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1)

    Wider hidden layers (64, 32) improve learning capacity over the
    original narrow (5, 5) design for this tabular regression task.
    """
    def __init__(self, input_dim: int):
        super(HousePriceANN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)          # Output layer — single value regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── 9. Initialise Model, Loss & Optimiser ────────────────────────────────────
INPUT_DIM  = X_train.shape[1]   # 5 features
EPOCHS     = 200
LR         = 1e-3               # Adam default; tune if needed

model     = HousePriceANN(input_dim=INPUT_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f"\nModel architecture:\n{model}")
total_params = sum(p.numel() for p in model.parameters())
print(f"Total trainable parameters: {total_params}")

# ── 10. Training Loop ─────────────────────────────────────────────────────────
train_losses = []
val_losses   = []
best_val_loss = float("inf")
CHECKPOINT   = "best_model.pt"

for epoch in range(EPOCHS):
    # ---- Training phase ----
    model.train()
    running_train_loss = 0.0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(X_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        running_train_loss += loss.item()

    epoch_train_loss = running_train_loss / len(train_loader)
    train_losses.append(epoch_train_loss)

    # ---- Validation phase ----
    model.eval()
    running_val_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            predictions = model(X_batch)
            running_val_loss += criterion(predictions, y_batch).item()

    epoch_val_loss = running_val_loss / len(test_loader)
    val_losses.append(epoch_val_loss)

    # Save best model checkpoint
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        torch.save(model.state_dict(), CHECKPOINT)

    if (epoch + 1) % 20 == 0:          # Print every 20 epochs to reduce noise
        print(f"Epoch [{epoch+1:>3}/{EPOCHS}] | "
              f"Train Loss: {epoch_train_loss:.4f} | "
              f"Val Loss:   {epoch_val_loss:.4f}")

print(f"\nBest validation loss: {best_val_loss:.4f} — checkpoint saved to '{CHECKPOINT}'")

# ── 11. Load Best Model & Evaluate ───────────────────────────────────────────
model.load_state_dict(torch.load(CHECKPOINT))
model.eval()

with torch.no_grad():
    train_preds = model(X_train_tensor)
    test_preds  = model(X_test_tensor)

    train_mse = criterion(train_preds, y_train_tensor).item()
    test_mse  = criterion(test_preds,  y_test_tensor).item()

print("\n── Final Evaluation ──")
print(f"Training MSE : {train_mse:.4f}  |  RMSE: {train_mse**0.5:.4f}")
print(f"Testing  MSE : {test_mse:.4f}  |  RMSE: {test_mse**0.5:.4f}")

# ── 12. Plot Training & Validation Loss Curves ───────────────────────────────
plt.figure(figsize=(9, 4))
plt.plot(train_losses, label="Train Loss", linewidth=1.8)
plt.plot(val_losses,   label="Val Loss",   linewidth=1.8, linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training vs. Validation Loss")
plt.legend()
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
plt.show()

# ── 13. Actual vs. Predicted Plot ─────────────────────────────────────────────
test_actual    = y_test_tensor.numpy().flatten()
test_predicted = test_preds.numpy().flatten()

plt.figure(figsize=(6, 6))
plt.scatter(test_actual, test_predicted, alpha=0.6, edgecolors='k', linewidths=0.4)
plt.plot([test_actual.min(), test_actual.max()],
         [test_actual.min(), test_actual.max()],
         'r--', linewidth=1.5, label="Perfect prediction")
plt.xlabel("Actual MEDV")
plt.ylabel("Predicted MEDV")
plt.title("Actual vs. Predicted House Prices (Test Set)")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=150)
plt.show()