import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import joblib
import matplotlib.pyplot as plt

DATA_FILE = "mdn_data/mdn_splits.npz"
SCALER_FILE = "mdn_data/Y_scaler.pkl"
CHECKPOINT_FILE = "mdn_run/best_mdn.pt"
OUTPUT_DIR = "mdn_distribution_plots"
SAMPLE_INDEX = 0
INPUT_DIM, TARGET_DIM, NUM_COMPONENTS = 8192, 6, 5
HIDDEN_1, HIDDEN_2 = 512, 256
PARAMETER_NAMES = [
    "Fibre length L (km)", "Dispersion beta2 (ps^2/km)",
    "Nonlinearity gamma (1/W/km)", "Attenuation alpha (dB/km)",
    "Peak power P0 (W)", "Pulse width T0 (ps)"
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
        self.class_head = nn.Linear(HIDDEN_2, 2)

    def forward(self, x):
        h = self.shared(x)
        b = x.shape[0]
        pi = self.pi_head(h)
        mu = self.mu_head(h).view(b, NUM_COMPONENTS, TARGET_DIM)
        log_sigma = self.log_sigma_head(h).view(b, NUM_COMPONENTS, TARGET_DIM)
        log_sigma = torch.clamp(log_sigma, -4.0, 1.5)
        return pi, mu, log_sigma, self.class_head(h)

print("Loading test data, scaler and trained checkpoint...")
data = np.load(DATA_FILE)
required = ["X_test", "Y_test", "pulse_test"]
missing = [k for k in required if k not in data.files]
if missing:
    raise KeyError(f"Missing {missing}; available keys: {data.files}")

X = data["X_test"].astype(np.float32)
Y_scaled = data["Y_test"].astype(np.float32)
pulse = data["pulse_test"].astype(int)
if not 0 <= SAMPLE_INDEX < len(X):
    raise IndexError(f"SAMPLE_INDEX must be between 0 and {len(X)-1}")

scaler = joblib.load(SCALER_FILE)
checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE, weights_only=False)
model = MDN().to(DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

with torch.no_grad():
    x = torch.from_numpy(X[SAMPLE_INDEX:SAMPLE_INDEX+1])
    pi_logits, mu_t, log_sigma_t, class_logits = model(x)
    weights = F.softmax(pi_logits, dim=-1).numpy()[0]
    means_scaled = mu_t.numpy()[0]
    sigmas_scaled = torch.exp(log_sigma_t).numpy()[0]
    predicted_class = int(class_logits.argmax(dim=1).item())

# Convert each component's means and standard deviations to physical units.
means = scaler.inverse_transform(means_scaled)
sigmas = sigmas_scaled / scaler.scale_[None, :]
actual = scaler.inverse_transform(Y_scaled[SAMPLE_INDEX:SAMPLE_INDEX+1])[0]
actual_class = pulse[SAMPLE_INDEX]

print("Checkpoint epoch:", checkpoint.get("epoch", "unknown"))
print("Test sample:", SAMPLE_INDEX)
print("Mixture weights:", np.round(weights, 4))
print("Actual targets:", np.round(actual, 6))
print("Actual pulse class:", actual_class, "| predicted:", predicted_class)

fig, axes = plt.subplots(2, 3, figsize=(17, 10))
for i, ax in enumerate(axes.ravel()):
    mus = means[:, i]
    sds = np.maximum(sigmas[:, i], 1e-12)
    lo = min(np.min(mus - 4*sds), actual[i])
    hi = max(np.max(mus + 4*sds), actual[i])
    grid = np.linspace(lo, hi, 1000)
    total = np.zeros_like(grid)

    for j in range(NUM_COMPONENTS):
        density = np.exp(-0.5*((grid-mus[j])/sds[j])**2) / (sds[j]*np.sqrt(2*np.pi))
        weighted = weights[j] * density
        total += weighted
        ax.plot(grid, weighted, "--", linewidth=1.5,
                label=f"Component {j+1}, weight={weights[j]:.2f}")

    ax.plot(grid, total, linewidth=2.8, label="Weighted mixture")
    ax.axvline(actual[i], linestyle=":", linewidth=2, label="Actual value")
    ax.axvline(np.sum(weights*mus), linestyle="-.", linewidth=1.5,
               label="Mixture mean")
    ax.set_title(PARAMETER_NAMES[i])
    ax.set_xlabel("Parameter value (physical units)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

fig.suptitle(
    f"MDN Gaussian Mixture Distributions — Test Sample {SAMPLE_INDEX}\n"
    f"Actual pulse class: {actual_class}; predicted pulse class: {predicted_class}",
    fontsize=14
)
fig.tight_layout(rect=[0, 0, 1, 0.94])
combined = os.path.join(OUTPUT_DIR, f"gaussian_mixture_sample_{SAMPLE_INDEX}.png")
fig.savefig(combined, dpi=300, bbox_inches="tight")
plt.show()

# Save each target's marginal mixture as a separate high-resolution figure.
for i, name in enumerate(PARAMETER_NAMES):
    fig1, ax = plt.subplots(figsize=(8, 5))
    mus, sds = means[:, i], np.maximum(sigmas[:, i], 1e-12)
    grid = np.linspace(min(np.min(mus-4*sds), actual[i]),
                       max(np.max(mus+4*sds), actual[i]), 1000)
    total = np.zeros_like(grid)
    for j in range(NUM_COMPONENTS):
        density = np.exp(-0.5*((grid-mus[j])/sds[j])**2)/(sds[j]*np.sqrt(2*np.pi))
        total += weights[j]*density
        ax.plot(grid, weights[j]*density, "--", label=f"Component {j+1} (w={weights[j]:.2f})")
    ax.plot(grid, total, linewidth=2.8, label="Weighted mixture")
    ax.axvline(actual[i], linestyle=":", linewidth=2, label="Actual value")
    ax.axvline(np.sum(weights*mus), linestyle="-.", label="Mixture mean")
    ax.set_title(f"{name} — test sample {SAMPLE_INDEX}")
    ax.set_xlabel("Parameter value (physical units)")
    ax.set_ylabel("Probability density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig1.tight_layout()
    fig1.savefig(os.path.join(OUTPUT_DIR, f"mixture_parameter_{i+1}.png"),
                 dpi=300, bbox_inches="tight")
    plt.close(fig1)

print("\nPlots saved in:", OUTPUT_DIR)
print("Combined plot:", combined)
print("Individual plots: mixture_parameter_1.png through mixture_parameter_6.png")
print("These are one-dimensional marginal distributions for one test sample, not a six-dimensional joint density.")
