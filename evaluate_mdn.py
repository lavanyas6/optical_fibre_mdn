import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report

DATA_FILE = "mdn_data/mdn_splits.npz"
SCALER_FILE = "mdn_data/Y_scaler.pkl"
CHECKPOINT_FILE = "mdn_run/best_mdn.pt"
OUTPUT_DIR = "mdn_independent_test_results"
INPUT_DIM, TARGET_DIM, NUM_COMPONENTS, NUM_CLASSES = 8192, 6, 5, 2
HIDDEN_1, HIDDEN_2 = 512, 256
BATCH_SIZE = 128
PARAMETER_NAMES = [
    "Length (km)", "Beta2 (ps^2/km)", "Gamma (1/W/km)",
    "Loss (dB/km)", "Peak power (W)", "Pulse width T0 (ps)"
]
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEVICE = torch.device("cpu")

class MDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN_1), nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2), nn.ReLU()
        )
        self.pi_head = nn.Linear(HIDDEN_2, NUM_COMPONENTS)
        self.mu_head = nn.Linear(HIDDEN_2, NUM_COMPONENTS * TARGET_DIM)
        self.log_sigma_head = nn.Linear(HIDDEN_2, NUM_COMPONENTS * TARGET_DIM)
        self.class_head = nn.Linear(HIDDEN_2, NUM_CLASSES)

    def forward(self, x):
        h = self.shared(x)
        b = x.shape[0]
        pi_logits = self.pi_head(h)
        mu = self.mu_head(h).view(b, NUM_COMPONENTS, TARGET_DIM)
        log_sigma = self.log_sigma_head(h).view(b, NUM_COMPONENTS, TARGET_DIM)
        log_sigma = torch.clamp(log_sigma, -4.0, 1.5)
        return pi_logits, mu, log_sigma, self.class_head(h)

print("Loading independent test data...")
data = np.load(DATA_FILE)
needed = ["X_test", "Y_test", "pulse_test"]
missing = [k for k in needed if k not in data.files]
if missing:
    raise KeyError(f"Missing {missing} in {DATA_FILE}. Available keys: {data.files}")

X = data["X_test"].astype(np.float32)
Y_scaled = data["Y_test"].astype(np.float32)
pulse_true = data["pulse_test"].astype(np.int64)
if X.ndim != 2 or X.shape[1] != INPUT_DIM:
    raise ValueError(f"Unexpected X_test shape: {X.shape}")
if Y_scaled.ndim != 2 or Y_scaled.shape[1] != TARGET_DIM:
    raise ValueError(f"Unexpected Y_test shape: {Y_scaled.shape}")
if not np.isfinite(X).all() or not np.isfinite(Y_scaled).all():
    raise ValueError("Test data contains NaN or infinite values.")
print("Independent test samples:", len(X))
print("Test input shape:", X.shape)

print("Loading target scaler and best checkpoint...")
scaler = joblib.load(SCALER_FILE)
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
if checkpoint.get("config", {}).get("covariance") not in (None, "diagonal"):
    raise ValueError("Checkpoint is not the expected diagonal-covariance model.")

model = MDN().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))

# Predict the conditional mixture mean and pulse class in batches.
pred_scaled, pred_classes = [], []
with torch.no_grad():
    for start in range(0, len(X), BATCH_SIZE):
        xb = torch.from_numpy(X[start:start+BATCH_SIZE]).to(DEVICE)
        pi_logits, mu, log_sigma, class_logits = model(xb)
        weights = F.softmax(pi_logits, dim=-1)
        mean = torch.sum(weights.unsqueeze(-1) * mu, dim=1)
        pred_scaled.append(mean.cpu().numpy())
        pred_classes.append(class_logits.argmax(dim=1).cpu().numpy())

Y_pred_scaled = np.concatenate(pred_scaled)
pulse_pred = np.concatenate(pred_classes)
Y_true = scaler.inverse_transform(Y_scaled)
Y_pred = scaler.inverse_transform(Y_pred_scaled)

# Continuous-parameter metrics, reported in original physical units.
rows = []
for i, name in enumerate(PARAMETER_NAMES):
    rows.append({
        "Parameter": name,
        "MAE": mean_absolute_error(Y_true[:, i], Y_pred[:, i]),
        "RMSE": np.sqrt(mean_squared_error(Y_true[:, i], Y_pred[:, i])),
        "R2": r2_score(Y_true[:, i], Y_pred[:, i])
    })
metrics = pd.DataFrame(rows)
metrics.to_csv(os.path.join(OUTPUT_DIR, "regression_metrics.csv"), index=False)
print("\nINDEPENDENT TEST REGRESSION METRICS")
print(metrics.to_string(index=False, float_format=lambda v: f"{v:.6g}"))

# Pulse classification metrics.
accuracy = accuracy_score(pulse_true, pulse_pred)
cm = confusion_matrix(pulse_true, pulse_pred, labels=[0, 1])
report = classification_report(
    pulse_true, pulse_pred, labels=[0, 1],
    target_names=["Gaussian (0)", "Sech (1)"], zero_division=0
)
print(f"\nIndependent test pulse accuracy: {accuracy*100:.3f}%")
print("Confusion matrix (rows=true, columns=predicted):\n", cm)
print(report)
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(f"Accuracy: {accuracy*100:.3f}%\n\n")
    f.write("Confusion matrix (rows=true, columns=predicted):\n")
    f.write(np.array2string(cm) + "\n\n" + report)

# Save actual and predicted values for every test sample.
out = {}
for i, name in enumerate(PARAMETER_NAMES):
    key = ["length_km", "beta2_ps2_km", "gamma_per_W_km",
           "loss_dB_km", "peak_power_W", "pulse_width_T0_ps"][i]
    out[f"actual_{key}"] = Y_true[:, i]
    out[f"predicted_{key}"] = Y_pred[:, i]
out["actual_pulse_type"] = pulse_true
out["predicted_pulse_type"] = pulse_pred
pd.DataFrame(out).to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

# Actual-versus-predicted plots.
for i, name in enumerate(PARAMETER_NAMES):
    plt.figure(figsize=(7, 6))
    plt.scatter(Y_true[:, i], Y_pred[:, i], alpha=0.35, s=10)
    lo = min(Y_true[:, i].min(), Y_pred[:, i].min())
    hi = max(Y_true[:, i].max(), Y_pred[:, i].max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel(f"Actual {name}")
    plt.ylabel(f"Predicted {name}")
    plt.title(f"Independent Test: Actual vs Predicted - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"actual_vs_predicted_{i+1}.png"), dpi=300)
    plt.close()

# Confusion matrix plot.
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest")
plt.title("Independent Test Pulse Classification")
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.xticks([0, 1], ["Gaussian", "Sech"])
plt.yticks([0, 1], ["Gaussian", "Sech"])
for r in range(2):
    for c in range(2):
        plt.text(c, r, str(cm[r, c]), ha="center", va="center")
plt.colorbar(label="Number of samples")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "pulse_confusion_matrix.png"), dpi=300)
plt.close()

print("\nIndependent test evaluation completed.")
print("All results saved in:", OUTPUT_DIR)
