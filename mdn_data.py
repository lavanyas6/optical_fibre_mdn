
import os
import h5py
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import joblib


# =====================================================
# CONFIGURATION
# =====================================================

DATASET_FILE = "nlse_scalar_dataset.h5"
OUTPUT_DIR = "mdn_data"

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =====================================================
# 1. LOAD DATASET
# =====================================================

print("\nLoading synthetic dataset...")

with h5py.File(DATASET_FILE, "r") as hf:

    input_temporal = hf["input_temporal_intensity"][:]
    input_spectral = hf["input_spectral_intensity"][:]

    output_temporal = hf["output_temporal_intensity"][:]
    output_spectral = hf["output_spectral_intensity"][:]

    labels = hf["meta_parameters"][:]


print("Dataset loaded successfully.")

print("Input temporal:", input_temporal.shape)
print("Input spectral:", input_spectral.shape)
print("Output temporal:", output_temporal.shape)
print("Output spectral:", output_spectral.shape)
print("Labels:", labels.shape)


# =====================================================
# 2. CONSTRUCT INPUT FEATURES
# =====================================================

print("\nConstructing input feature matrix...")

X = np.concatenate(
    [
        input_temporal,
        input_spectral,
        output_temporal,
        output_spectral
    ],
    axis=1
)

print("X shape:", X.shape)


# =====================================================
# 3. CONSTRUCT TARGET VARIABLES
# =====================================================

# Original label columns:
# 0 = L
# 1 = beta2
# 2 = D
# 3 = gamma
# 4 = loss
# 5 = P0
# 6 = T0
# 7 = pulse type

# Exclude D because it is derived from beta2.

Y_continuous = labels[:, [0, 1, 3, 4, 5, 6]]

Y_categorical = labels[:, 7].astype(np.int64)

target_names = [
    "L_km",
    "beta2_ps2_km",
    "gamma_W_km",
    "loss_dB_km",
    "P0_W",
    "T0_ps"
]

print("Continuous targets:", Y_continuous.shape)
print("Categorical targets:", Y_categorical.shape)


# =====================================================
# 4. TRAIN / VALIDATION / TEST SPLIT
# =====================================================

print("\nSplitting dataset...")

indices = np.arange(len(X))

train_idx, temporary_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=Y_categorical
)

val_idx, test_idx = train_test_split(
    temporary_idx,
    test_size=2/3,
    random_state=RANDOM_STATE,
    stratify=Y_categorical[temporary_idx]
)

print("Training samples:", len(train_idx))
print("Validation samples:", len(val_idx))
print("Testing samples:", len(test_idx))


# =====================================================
# 5. NORMALIZE INPUT FEATURES
# =====================================================

print("\nNormalizing input features...")

# Fit only on training samples to prevent data leakage.

X_scaler = StandardScaler()

X_train = X_scaler.fit_transform(X[train_idx])
X_val = X_scaler.transform(X[val_idx])
X_test = X_scaler.transform(X[test_idx])


# =====================================================
# 6. NORMALIZE CONTINUOUS TARGETS
# =====================================================

print("Normalizing continuous targets...")

Y_scaler = MinMaxScaler(feature_range=(-1, 1))

Y_train = Y_scaler.fit_transform(
    Y_continuous[train_idx]
)

Y_val = Y_scaler.transform(
    Y_continuous[val_idx]
)

Y_test = Y_scaler.transform(
    Y_continuous[test_idx]
)


# =====================================================
# 7. CATEGORICAL TARGETS
# =====================================================

pulse_train = Y_categorical[train_idx]
pulse_val = Y_categorical[val_idx]
pulse_test = Y_categorical[test_idx]


# =====================================================
# 8. SAVE PREPARED DATA
# =====================================================

print("\nSaving prepared dataset...")

np.savez_compressed(
    os.path.join(OUTPUT_DIR, "mdn_splits.npz"),

    X_train=X_train.astype(np.float32),
    X_val=X_val.astype(np.float32),
    X_test=X_test.astype(np.float32),

    Y_train=Y_train.astype(np.float32),
    Y_val=Y_val.astype(np.float32),
    Y_test=Y_test.astype(np.float32),

    pulse_train=pulse_train,
    pulse_val=pulse_val,
    pulse_test=pulse_test,

    train_idx=train_idx,
    val_idx=val_idx,
    test_idx=test_idx
)


# =====================================================
# 9. SAVE SCALERS
# =====================================================

joblib.dump(
    X_scaler,
    os.path.join(OUTPUT_DIR, "X_scaler.pkl")
)

joblib.dump(
    Y_scaler,
    os.path.join(OUTPUT_DIR, "Y_scaler.pkl")
)


# =====================================================
# 10. FINAL SUMMARY
# =====================================================

print("\n" + "=" * 55)
print("MDN DATA PREPARATION COMPLETED")
print("=" * 55)

print("Input dimensions:", X_train.shape[1])
print("Continuous outputs:", Y_train.shape[1])

print("Training:", X_train.shape)
print("Validation:", X_val.shape)
print("Testing:", X_test.shape)

print("\nTarget order:")

for i, name in enumerate(target_names):
    print(f"{i}: {name}")

print("\nFiles saved in:", OUTPUT_DIR)
print("=" * 55)
