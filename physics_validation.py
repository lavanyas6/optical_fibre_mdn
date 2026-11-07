
import h5py
import numpy as np

FILE_NAME = "nlse_scalar_dataset.h5"

with h5py.File(FILE_NAME, "r") as hf:

    labels = hf["meta_parameters"][:]

    input_t = hf["input_temporal_intensity"]
    output_t = hf["output_temporal_intensity"]

    input_w = hf["input_spectral_intensity"]
    output_w = hf["output_spectral_intensity"]

    # Select the first Gaussian sample
    idx = np.where(labels[:, 7] == 0)[0][0]

    L, beta2, D, gamma, loss, P0, T0, pulse = labels[idx]

    print("\nSELECTED SAMPLE PARAMETERS")
    print("-" * 40)

    print(f"Length       : {L:.5f} km")
    print(f"Beta2        : {beta2:.5f} ps^2/km")
    print(f"Gamma        : {gamma:.5f} W^-1 km^-1")
    print(f"Loss         : {loss:.5f} dB/km")
    print(f"Peak power   : {P0:.5f} W")
    print(f"Pulse width  : {T0:.5f} ps")

    print("\nTEMPORAL INTENSITY")
    print("-" * 40)

    print("Input peak :", input_t[idx].max())
    print("Output peak:", output_t[idx].max())

    print("\nSPECTRAL INTENSITY")
    print("-" * 40)

    print("Input spectral peak :", input_w[idx].max())
    print("Output spectral peak:", output_w[idx].max())

    print("\nPOWER TRANSMISSION CHECK")
    print("-" * 40)

    measured = (
        np.trapezoid(output_t[idx])
        / np.trapezoid(input_t[idx])
    )

    expected = 10 ** (-loss * L / 10)

    print(f"Measured transmission: {measured:.6f}")
    print(f"Expected transmission: {expected:.6f}")
