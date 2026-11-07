import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Configuration
DATA_FILE = "mdn_data/mdn_splits.npz"
OUTPUT_DIR = "mdn_run"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42
INPUT_DIM = 8192
TARGET_DIM = 6
NUM_COMPONENTS = 5
NUM_CLASSES = 2
HIDDEN_1 = 512
HIDDEN_2 = 256
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-4
PATIENCE = 12
CLASSIFICATION_WEIGHT = 0.2
MIN_LOG_SIGMA = -4.0
MAX_LOG_SIGMA = 1.5
INITIAL_SIGMA = 0.3
GRADIENT_CLIP_NORM = 1.0

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cpu")
print("Using device:", DEVICE)

# Load and validate prepared data
print("\nLoading prepared MDN data...")
data = np.load(DATA_FILE)
X_train = data["X_train"].astype(np.float32)
X_val = data["X_val"].astype(np.float32)
Y_train = data["Y_train"].astype(np.float32)
Y_val = data["Y_val"].astype(np.float32)
pulse_train = data["pulse_train"].astype(np.int64)
pulse_val = data["pulse_val"].astype(np.int64)

if X_train.ndim != 2 or X_train.shape[1] != INPUT_DIM:
    raise ValueError(f"Expected X_train shape (N, {INPUT_DIM}), got {X_train.shape}")
if X_val.ndim != 2 or X_val.shape[1] != INPUT_DIM:
    raise ValueError(f"Expected X_val shape (N, {INPUT_DIM}), got {X_val.shape}")
if Y_train.ndim != 2 or Y_train.shape[1] != TARGET_DIM:
    raise ValueError(f"Expected Y_train shape (N, {TARGET_DIM}), got {Y_train.shape}")
if Y_val.ndim != 2 or Y_val.shape[1] != TARGET_DIM:
    raise ValueError(f"Expected Y_val shape (N, {TARGET_DIM}), got {Y_val.shape}")

for name, arr in {"X_train": X_train, "X_val": X_val,
                  "Y_train": Y_train, "Y_val": Y_val}.items():
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
if not np.isin(pulse_train, [0, 1]).all() or not np.isin(pulse_val, [0, 1]).all():
    raise ValueError("Pulse labels must be 0 or 1.")

print("Training inputs:", X_train.shape)
print("Validation inputs:", X_val.shape)
print("Training targets:", Y_train.shape)
print("Validation targets:", Y_val.shape)

train_dataset = TensorDataset(torch.from_numpy(X_train),
                              torch.from_numpy(Y_train),
                              torch.from_numpy(pulse_train))
val_dataset = TensorDataset(torch.from_numpy(X_val),
                            torch.from_numpy(Y_val),
                            torch.from_numpy(pulse_val))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

class MDN(nn.Module):
    """Five-component MDN with diagonal Gaussian covariance."""
    def __init__(self, input_dim, target_dim, components, num_classes):
        super().__init__()
        self.target_dim = target_dim
        self.components = components
        self.shared = nn.Sequential(
            nn.Linear(input_dim, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU()
        )
        self.pi_head = nn.Linear(HIDDEN_2, components)
        self.mu_head = nn.Linear(HIDDEN_2, components * target_dim)
        self.log_sigma_head = nn.Linear(HIDDEN_2, components * target_dim)
        self.class_head = nn.Linear(HIDDEN_2, num_classes)
        nn.init.constant_(self.log_sigma_head.bias, np.log(INITIAL_SIGMA))
        nn.init.normal_(self.log_sigma_head.weight, mean=0.0, std=0.005)

    def forward(self, x):
        features = self.shared(x)
        batch = x.shape[0]
        pi_logits = self.pi_head(features)
        mu = self.mu_head(features).view(batch, self.components, self.target_dim)
        log_sigma = self.log_sigma_head(features).view(
            batch, self.components, self.target_dim)
        log_sigma = torch.clamp(log_sigma, MIN_LOG_SIGMA, MAX_LOG_SIGMA)
        class_logits = self.class_head(features)
        return pi_logits, mu, log_sigma, class_logits

def mdn_loss(target, pi_logits, mu, log_sigma):
    target = target.unsqueeze(1)
    inv_var = torch.exp(-2.0 * log_sigma)
    mahalanobis = torch.sum((target - mu).pow(2) * inv_var, dim=-1)
    log_det = 2.0 * torch.sum(log_sigma, dim=-1)
    constant = TARGET_DIM * np.log(2.0 * np.pi)
    log_gaussian = -0.5 * (constant + log_det + mahalanobis)
    log_pi = F.log_softmax(pi_logits, dim=-1)
    log_probability = torch.logsumexp(log_pi + log_gaussian, dim=1)
    return -log_probability.mean()

model = MDN(INPUT_DIM, TARGET_DIM, NUM_COMPONENTS, NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, eps=1e-8)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=4, min_lr=1e-6)

def train_one_epoch():
    model.train()
    sums = np.zeros(3, dtype=np.float64)
    count = 0
    for x, y, pulse in train_loader:
        x, y, pulse = x.to(DEVICE), y.to(DEVICE), pulse.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        pi_logits, mu, log_sigma, class_logits = model(x)
        nll = mdn_loss(y, pi_logits, mu, log_sigma)
        class_loss = F.cross_entropy(class_logits, pulse)
        loss = nll + CLASSIFICATION_WEIGHT * class_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss; stopped safely.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP_NORM)
        optimizer.step()
        n = x.shape[0]
        sums += [loss.item() * n, nll.item() * n, class_loss.item() * n]
        count += n
    return tuple(sums / count)

@torch.no_grad()
def validate():
    model.eval()
    sums = np.zeros(3, dtype=np.float64)
    correct = count = 0
    for x, y, pulse in val_loader:
        x, y, pulse = x.to(DEVICE), y.to(DEVICE), pulse.to(DEVICE)
        pi_logits, mu, log_sigma, class_logits = model(x)
        nll = mdn_loss(y, pi_logits, mu, log_sigma)
        class_loss = F.cross_entropy(class_logits, pulse)
        loss = nll + CLASSIFICATION_WEIGHT * class_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss; stopped safely.")
        n = x.shape[0]
        sums += [loss.item() * n, nll.item() * n, class_loss.item() * n]
        correct += (class_logits.argmax(dim=1) == pulse).sum().item()
        count += n
    return (*tuple(sums / count), correct / count)

print("\nStarting stable diagonal-covariance MDN training...")
print("=" * 78)
best_val_loss = float("inf")
best_epoch = 0
patience_counter = 0
history = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_nll, train_class = train_one_epoch()
    val_loss, val_nll, val_class, val_accuracy = validate()
    scheduler.step(val_loss)
    lr = optimizer.param_groups[0]["lr"]
    history.append([epoch, train_loss, train_nll, train_class,
                    val_loss, val_nll, val_class, val_accuracy, lr])
    print(f"Epoch {epoch:03d}/{EPOCHS} | Train: {train_loss:.5f} | "
          f"Val: {val_loss:.5f} | Val NLL: {val_nll:.5f} | "
          f"Pulse accuracy: {val_accuracy*100:.2f}% | LR: {lr:.2e}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        patience_counter = 0
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "input_dim": INPUT_DIM, "target_dim": TARGET_DIM,
                "components": NUM_COMPONENTS, "num_classes": NUM_CLASSES,
                "covariance": "diagonal",
                "min_log_sigma": MIN_LOG_SIGMA,
                "max_log_sigma": MAX_LOG_SIGMA
            }
        }, os.path.join(OUTPUT_DIR, "best_mdn.pt"))
    else:
        patience_counter += 1

    if patience_counter >= PATIENCE:
        print("\nEarly stopping triggered.")
        break

np.savetxt(
    os.path.join(OUTPUT_DIR, "training_history.csv"),
    np.asarray(history, dtype=np.float64),
    delimiter=",",
    header=("epoch,train_loss,train_nll,train_class_loss,"
            "val_loss,val_nll,val_class_loss,val_accuracy,learning_rate"),
    comments=""
)

print("\n" + "=" * 78)
print("MDN TRAINING COMPLETED")
print("=" * 78)
print("Best epoch:", best_epoch)
print("Best validation loss:", best_val_loss)
print("Saved model:", os.path.join(OUTPUT_DIR, "best_mdn.pt"))
print("Saved history:", os.path.join(OUTPUT_DIR, "training_history.csv"))
print("=" * 78)
