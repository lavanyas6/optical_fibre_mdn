import multiprocessing as mp
import os
import h5py
import numpy as np
from scipy.stats import qmc

# Standard optical constants for dispersion mapping at 1550 nm
C_M_S = 299792458.0  # Speed of light (m/s)
LAMBDA_NM = 1550.0  # Wavelength (nm)


def beta2_to_D(beta2_ps2_km, wavelength_nm=LAMBDA_NM):
    """Converts GVD parameter beta2 (ps^2/km) to Dispersion parameter D (ps/(nm*km))."""
    wavelength_m = wavelength_nm * 1e-9
    # Formula: D = - (2 * pi * c / lambda^2) * beta2
    conversion_factor = -(2.0 * np.pi * C_M_S / (wavelength_m**2)) * 1e-21
    return beta2_ps2_km * conversion_factor


# --- 1. WORKER SIMULATION FUNCTION ---
def run_ssfm_simulation(args):
    sample_id, params = args
    L_km, beta2, gamma, loss_dB, P0, T0, pulse_type_flag = params

    # Pulse Shape Decision: 0 = Gaussian, 1 = Hyperbolic Secant (sech)
    pulse_type = 0 if pulse_type_flag < 0.5 else 1

    # Grid setup
    N = 2048
    T_max = 800.0
    dt = T_max / N
    t = np.linspace(-T_max / 2, T_max / 2 - dt, N)
    omega = 2 * np.pi * np.fft.fftfreq(N, d=dt)

    alpha = loss_dB / 4.343

    # Construct input pulse profile based on sampled parameters
    if pulse_type == 0:
        A_in = np.sqrt(P0) * np.exp(-0.5 * (t / T0) ** 2).astype(np.complex128)
    else:
        A_in = (np.sqrt(P0) / np.cosh(t / T0)).astype(np.complex128)

    input_t_intensity = np.abs(A_in) ** 2
    input_w_intensity = np.abs(np.fft.fftshift(np.fft.fft(A_in))) ** 2

    # SSFM Propagation Setup
    num_steps = 200
    dz = L_km / num_steps

    half_dispersion = np.exp(
        (-alpha / 2.0 - 1j * (beta2 / 2.0) * (omega**2)) * (dz / 2.0)
    )
    full_dispersion = np.exp(
        (-alpha / 2.0 - 1j * (beta2 / 2.0) * (omega**2)) * dz
    )

    A = A_in.copy()
    A_fft = np.fft.fft(A)
    A = np.fft.ifft(A_fft * half_dispersion)

    for _ in range(num_steps - 1):
        N_op = np.exp(1j * gamma * np.abs(A) ** 2 * dz)
        A = A * N_op
        A_fft = np.fft.fft(A)
        A = np.fft.ifft(A_fft * full_dispersion)

    N_op = np.exp(1j * gamma * np.abs(A) ** 2 * dz)
    A = A * N_op
    A_fft = np.fft.fft(A)
    A = np.fft.ifft(A_fft * half_dispersion)

    # Raw output intensities
    output_t_intensity = np.abs(A) ** 2
    output_w_intensity = np.abs(np.fft.fftshift(np.fft.fft(A))) ** 2

    # Normalization per sample
    input_t_norm = (input_t_intensity / P0).astype(np.float32)
    output_t_norm = (output_t_intensity / P0).astype(np.float32)

    max_in_w = np.max(input_w_intensity)
    input_w_norm = (input_w_intensity / max_in_w).astype(np.float32)
    output_w_norm = (output_w_intensity / max_in_w).astype(np.float32)

    # Compute explicit Dispersion parameter D
    D_dispersion = beta2_to_D(beta2)

    # Label vector: [L_km, beta2, D, gamma, loss_dB, P0, T0, pulse_type]
    labels = np.array(
        [L_km, beta2, D_dispersion, gamma, loss_dB, P0, T0, pulse_type],
        dtype=np.float32,
    )

    return (
        sample_id,
        input_t_norm,
        input_w_norm,
        output_t_norm,
        output_w_norm,
        labels,
        t.astype(np.float32),
    )


# --- 2. STREAMED DATASET GENERATOR ---
def generate_dataset(
    num_samples=50000, output_filename="nlse_tnn_dataset.h5"
):
    print(
        f"--- Generating {num_samples} Samples using Streamed Multiprocessing ---"
    )

    # 7D Latin Hypercube Sampling:
    # [L_km, beta2, gamma, loss_dB, P0 (W), T0 (ps), pulse_type_flag]
    bounds_lower = [10.0, -25.0, 1.0, 0.18, 0.01, 10.0, 0.0]
    bounds_upper = [80.0, -5.0, 5.0, 0.35, 0.10, 30.0, 1.0]

    sampler = qmc.LatinHypercube(d=7, seed=42)
    sample_points = sampler.random(n=num_samples)
    scaled_params = qmc.scale(sample_points, bounds_lower, bounds_upper)

    N = 2048
    input_temporal = np.zeros((num_samples, N), dtype=np.float32)
    input_spectral = np.zeros((num_samples, N), dtype=np.float32)
    output_temporal = np.zeros((num_samples, N), dtype=np.float32)
    output_spectral = np.zeros((num_samples, N), dtype=np.float32)
    parameter_labels = np.zeros((num_samples, 8), dtype=np.float32)

    time_vector = None
    num_workers = max(1, (os.cpu_count() or 4) - 2)
    print(f"Utilizing {num_workers} CPU worker threads...")

    tasks = [(i, scaled_params[i]) for i in range(num_samples)]

    with mp.Pool(processes=num_workers) as pool:
        completed = 0
        for result in pool.imap_unordered(
            run_ssfm_simulation, tasks, chunksize=100
        ):
            idx, in_t, in_w, out_t, out_w, labels, t_axis = result

            input_temporal[idx, :] = in_t
            input_spectral[idx, :] = in_w
            output_temporal[idx, :] = out_t
            output_spectral[idx, :] = out_w
            parameter_labels[idx] = labels

            if time_vector is None:
                time_vector = t_axis

            completed += 1
            if completed % 5000 == 0 or completed == num_samples:
                print(
                    f"Progress: {completed}/{num_samples} samples generated ({completed/num_samples*100:.1f}%)"
                )

    print("\nWriting compressed dataset to HDF5 file...")
    with h5py.File(output_filename, "w") as hf:
        hf.create_dataset(
            "input_temporal_intensity", data=input_temporal, compression="gzip"
        )
        hf.create_dataset(
            "input_spectral_intensity", data=input_spectral, compression="gzip"
        )
        hf.create_dataset(
            "output_temporal_intensity",
            data=output_temporal,
            compression="gzip",
        )
        hf.create_dataset(
            "output_spectral_intensity",
            data=output_spectral,
            compression="gzip",
        )
        hf.create_dataset(
            "meta_parameters", data=parameter_labels, compression="gzip"
        )
        hf.create_dataset("time_vector", data=time_vector, compression="gzip")

        # Column definitions attached as HDF5 metadata attributes
        hf["meta_parameters"].attrs["column_names"] = [
            "L_km",
            "beta2_ps2_km",
            "D_ps_nm_km",
            "gamma_W_km",
            "loss_dB_km",
            "P0_W",
            "T0_ps",
            "pulse_type_0G_1S",
        ]

    print(
        f"SUCCESS: {num_samples} samples compiled and saved to '{output_filename}'!"
    )


if __name__ == "__main__":
    generate_dataset(num_samples=50000)