
import h5py
import numpy as np
import os

FILE_NAME = "nlse_scalar_dataset.h5"

print("=" * 55)
print("       SYNTHETIC DATASET INSPECTION")
print("=" * 55)

print(f"\nFile exists: {os.path.exists(FILE_NAME)}")

with h5py.File(FILE_NAME, "r") as hf:

    print("\nDATASET STRUCTURE")
    print("-" * 40)

    for name in hf.keys():
        data = hf[name]
        print(f"{name:30s} Shape: {data.shape}")

    print("\nPARAMETER INFORMATION")
    print("-" * 40)

    labels = hf["meta_parameters"][:]

    names = hf["meta_parameters"].attrs["column_names"]

    for i, name in enumerate(names):
        name = name.decode() if isinstance(name, bytes) else name
        print(f"{name:25s} Min: {labels[:, i].min():.5f} "
              f"Max: {labels[:, i].max():.5f}")

    print("\nDATA QUALITY CHECK")
    print("-" * 40)

    for name in hf.keys():
        data = hf[name]

        if data.dtype.kind in "fiu":
            arr = data[:]
            print(
                f"{name:30s} "
                f"NaN: {np.isnan(arr).sum()} "
                f"Inf: {np.isinf(arr).sum()}"
            )

    print("\nSUMMARY")
    print("-" * 40)
    print("Number of samples:", labels.shape[0])
    print("Number of labels:", labels.shape[1])
    print("All checks completed.")

print("\n" + "=" * 55)
