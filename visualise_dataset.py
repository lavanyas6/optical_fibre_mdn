
import h5py
import numpy as np
import matplotlib.pyplot as plt

FILE_NAME = "nlse_scalar_dataset.h5"

with h5py.File(FILE_NAME, "r") as hf:

    time = hf["time_vector_ps"][:]
    frequency = hf["frequency_vector_per_ps"][:]

    input_t = hf["input_temporal_intensity"]
    output_t = hf["output_temporal_intensity"]

    input_w = hf["input_spectral_intensity"]
    output_w = hf["output_spectral_intensity"]

    labels = hf["meta_parameters"][:]

    # Select one Gaussian and one sech sample
    gaussian_idx = np.where(labels[:, 7] == 0)[0][0]
    sech_idx = np.where(labels[:, 7] == 1)[0][0]

    for idx, pulse_name in [
        (gaussian_idx, "Gaussian"),
        (sech_idx, "Sech")
    ]:

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))

        # Input temporal intensity
        axes[0, 0].plot(time, input_t[idx])
        axes[0, 0].set_title(f"{pulse_name}: Input temporal intensity")
        axes[0, 0].set_xlabel("Time (ps)")
        axes[0, 0].set_ylabel("Normalized intensity")

        # Output temporal intensity
        axes[0, 1].plot(time, output_t[idx])
        axes[0, 1].set_title(f"{pulse_name}: Output temporal intensity")
        axes[0, 1].set_xlabel("Time (ps)")
        axes[0, 1].set_ylabel("Normalized intensity")

        # Input spectral intensity
        axes[1, 0].plot(frequency, input_w[idx])
        axes[1, 0].set_title(f"{pulse_name}: Input spectrum")
        axes[1, 0].set_xlabel("Frequency offset (1/ps)")
        axes[1, 0].set_ylabel("Normalized spectral intensity")

        # Output spectral intensity
        axes[1, 1].plot(frequency, output_w[idx])
        axes[1, 1].set_title(f"{pulse_name}: Output spectrum")
        axes[1, 1].set_xlabel("Frequency offset (1/ps)")
        axes[1, 1].set_ylabel("Normalized spectral intensity")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{pulse_name.lower()}_visualization.png", dpi=200)
        plt.show()
