
import numpy as np
import os

FILE = "mdn_data/mdn_splits.npz"

print("=" * 55)
print("         FINAL MDN DATA VALIDATION")
print("=" * 55)

data = np.load(FILE)

for name in data.files:
    arr = data[name]

    print(f"\n{name}:")
    print("Shape:", arr.shape)

    if arr.dtype.kind in "fiu":
        print("NaN:", np.isnan(arr).sum())
        print("Infinity:", np.isinf(arr).sum())

# Check pulse-type balance
print("\nPULSE TYPE DISTRIBUTION")
print("-" * 40)

for split in ["train", "val", "test"]:

    labels = data[f"pulse_{split}"]

    gaussian = np.sum(labels == 0)
    sech = np.sum(labels == 1)

    print(f"\n{split.upper()}")
    print("Gaussian:", gaussian)
    print("Sech:", sech)
    print("Total:", len(labels))

# Check expected dimensions
print("\nDIMENSION CHECK")
print("-" * 40)

assert data["X_train"].shape == (35000, 8192)
assert data["X_val"].shape == (7500, 8192)
assert data["X_test"].shape == (7500, 8192)

assert data["Y_train"].shape == (35000, 6)
assert data["Y_val"].shape == (7500, 6)
assert data["Y_test"].shape == (7500, 6)

# Check numerical validity
for name in [
    "X_train", "X_val", "X_test",
    "Y_train", "Y_val", "Y_test"
]:
    assert np.isfinite(data[name]).all(), (
        f"Invalid values found in {name}"
    )

# Check categorical labels
for name in ["pulse_train", "pulse_val", "pulse_test"]:
    assert set(np.unique(data[name])).issubset({0, 1})

print("\nALL ASSERTIONS PASSED")
print("MDN DATA IS READY FOR THE NEXT STAGE.")
print("=" * 55)
