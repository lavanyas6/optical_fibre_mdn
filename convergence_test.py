
import h5py
import numpy as np
import matplotlib.pyplot as plt

from datasetgenerator import (
    make_input_pulse,
    propagate_ssfm_strang
)

FILE_NAME = "nlse_scalar_dataset.h5"

# Different numbers of propagation steps
step_counts = [100, 200, 400, 800, 1600]

N = 2048
T_MAX = 800.0

dt = T_MAX / N
t = (np.arange(N) - N // 2) * dt

omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)


def run_convergence_test(sample_id, pulse_name, labels):

    L, beta2, D, gamma, loss, P0, T0, pulse_type = labels

    A_in = make_input_pulse(t, P0, T0, int(pulse_type))

    results = {}

    for steps in step_counts:

        A_out = propagate_ssfm_strang(
            A_in,
            omega,
            L,
            beta2,
            gamma,
            loss,
            steps
        )

        results[steps] = A_out

    # Highest-resolution result used as reference
    reference = results[1600]

    print(f"\n{pulse_name} PULSE")
    print("-" * 55)
    print("Steps       Relative L2 error       Peak intensity")

    errors = []

    for steps in step_counts:

        A = results[steps]

        error = (
            np.linalg.norm(A - reference)
            / np.linalg.norm(reference)
        )

        peak = np.max(np.abs(A) ** 2)

        errors.append(error)

        print(f"{steps:<12}{error:<24.8e}{peak:.8e}")

    # Plot convergence errors
    plt.figure(figsize=(8, 5))

    plt.loglog(
        step_counts[:-1],
        errors[:-1],
        marker="o"
    )

    plt.xlabel("Number of SSFM steps")
    plt.ylabel("Relative complex-field L2 error")
    plt.title(f"{pulse_name} pulse convergence")

    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        f"{pulse_name.lower()}_convergence.png",
        dpi=200
    )

    plt.show()


with h5py.File(FILE_NAME, "r") as hf:

    labels = hf["meta_parameters"][:]

    gaussian_id = np.where(labels[:, 7] == 0)[0][0]
    sech_id = np.where(labels[:, 7] == 1)[0][0]

    gaussian_params = labels[gaussian_id]
    sech_params = labels[sech_id]


print("STARTING SSFM CONVERGENCE TEST")

run_convergence_test(
    gaussian_id,
    "Gaussian",
    gaussian_params
)

run_convergence_test(
    sech_id,
    "Sech",
    sech_params
)

print("\nCONVERGENCE TEST COMPLETED")
