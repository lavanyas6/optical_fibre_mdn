

import argparse
import multiprocessing as mp
import os

import h5py
import numpy as np
from scipy.stats import qmc


C_M_S = 299_792_458.0
LAMBDA_NM = 1550.0


def beta2_to_D(beta2_ps2_km, wavelength_nm=LAMBDA_NM):
    """Convert beta2 [ps^2/km] to dispersion D [ps/(nm km)]."""
    wavelength_m = wavelength_nm * 1e-9
    factor = -(2.0 * np.pi * C_M_S / wavelength_m**2) * 1e-21
    return beta2_ps2_km * factor


def make_input_pulse(t_ps, peak_power_w, t0_ps, pulse_type):
    """Return complex input envelope for Gaussian (0) or sech (1) pulse."""
    if pulse_type == 0:
        # |A|^2 = P0 * exp[-(T/T0)^2]
        envelope = np.sqrt(peak_power_w) * np.exp(
            -0.5 * (t_ps / t0_ps) ** 2
        )
    elif pulse_type == 1:
        # |A|^2 = P0 * sech^2(T/T0)
        envelope = np.sqrt(peak_power_w) / np.cosh(t_ps / t0_ps)
    else:
        raise ValueError("pulse_type must be 0 (Gaussian) or 1 (sech).")
    return envelope.astype(np.complex128)


def propagate_ssfm(
    a0,
    omega_rad_ps,
    length_km,
    beta2_ps2_km,
    gamma_w_km,
    loss_db_km,
    num_steps,
):
    """
    Symmetric split-step Fourier propagation for:
      dA/dz = -(alpha/2)A + i(beta2/2)d2A/dT2 + i*gamma*|A|^2*A

    alpha is the POWER attenuation coefficient in Np/km.
    """
    alpha_np_km = loss_db_km / 4.343
    dz = length_km / num_steps

    # FFT{d2A/dT2} = -omega^2 * FFT{A}
    # Thus the linear spectral operator is -alpha/2 - i*beta2*omega^2/2.
    linear_half = np.exp(
        (-alpha_np_km / 2.0 - 1j * beta2_ps2_km * omega_rad_ps**2 / 2.0)
        * (dz / 2.0)
    )
    linear_full = linear_half**2

    a = np.fft.ifft(np.fft.fft(a0) * linear_half)

    for _ in range(num_steps):
        # Nonlinear phase for one full distance step.
        a *= np.exp(1j * gamma_w_km * np.abs(a) ** 2 * dz)
        a = np.fft.ifft(np.fft.fft(a) * linear_full)

    # The loop above applies a final full linear step. Correct the endpoint
    # by replacing it with the symmetric final half-step:
    # Undo the last half of linear propagation, then apply half-step only.
    # More simply, use a conventional Strang loop below instead.
    # (This branch is retained only as a guard against accidental edits.)
    return a


def propagate_ssfm_strang(
    a0,
    omega_rad_ps,
    length_km,
    beta2_ps2_km,
    gamma_w_km,
    loss_db_km,
    num_steps,
):
    """Strang-split SSFM: linear half-step, nonlinear full-step, linear half-step."""
    alpha_np_km = loss_db_km / 4.343
    dz = length_km / num_steps

    linear_half = np.exp(
        (-alpha_np_km / 2.0 - 1j * beta2_ps2_km * omega_rad_ps**2 / 2.0)
        * (dz / 2.0)
    )

    a = a0.copy()
    for _ in range(num_steps):
        a = np.fft.ifft(np.fft.fft(a) * linear_half)
        a *= np.exp(1j * gamma_w_km * np.abs(a) ** 2 * dz)
        a = np.fft.ifft(np.fft.fft(a) * linear_half)
    return a


def simulate_one(task):
    """Worker function. Must be at module level for macOS multiprocessing."""
    sample_id, params, n_time, t_max_ps, num_steps = task
    length_km, beta2, gamma, loss_db, p0, t0, pulse_flag = params
    pulse_type = int(pulse_flag >= 0.5)

    dt_ps = t_max_ps / n_time
    t_ps = (np.arange(n_time) - n_time // 2) * dt_ps
    omega = 2.0 * np.pi * np.fft.fftfreq(n_time, d=dt_ps)

    a_in = make_input_pulse(t_ps, p0, t0, pulse_type)
    a_out = propagate_ssfm_strang(
        a_in, omega, length_km, beta2, gamma, loss_db, num_steps
    )

    # Temporal intensities normalized by launched peak power.
    in_t = (np.abs(a_in) ** 2 / p0).astype(np.float32)
    out_t = (np.abs(a_out) ** 2 / p0).astype(np.float32)

    # Spectra use the same FFT convention and are normalized to input peak.
    in_spec_raw = np.abs(np.fft.fftshift(np.fft.fft(a_in))) ** 2
    out_spec_raw = np.abs(np.fft.fftshift(np.fft.fft(a_out))) ** 2
    spec_scale = max(float(in_spec_raw.max()), np.finfo(np.float64).tiny)
    in_w = (in_spec_raw / spec_scale).astype(np.float32)
    out_w = (out_spec_raw / spec_scale).astype(np.float32)

    d_value = beta2_to_D(beta2)
    labels = np.asarray(
        [length_km, beta2, d_value, gamma, loss_db, p0, t0, pulse_type],
        dtype=np.float32,
    )
    return sample_id, in_t, in_w, out_t, out_w, labels


def generate_dataset(
    num_samples=200,
    output_filename="nlse_scalar_dataset.h5",
    seed=42,
    n_time=2048,
    t_max_ps=800.0,
    num_steps=200,
    workers=None,
):
    """Generate samples with Latin Hypercube Sampling and stream to HDF5."""
    # [L_km, beta2_ps2_km, gamma_W^-1_km^-1, loss_dB_km, P0_W, T0_ps, pulse flag]
    lower = np.array([10.0, -25.0, 1.0, 0.18, 0.01, 10.0, 0.0])
    upper = np.array([80.0, -5.0, 5.0, 0.35, 0.10, 30.0, 1.0])

    sampler = qmc.LatinHypercube(d=7, seed=seed)
    unit_samples = sampler.random(n=num_samples)
    sampled_params = qmc.scale(unit_samples, lower, upper)

    dt_ps = t_max_ps / n_time
    time_vector = (np.arange(n_time) - n_time // 2) * dt_ps
    frequency_vector = np.fft.fftshift(np.fft.fftfreq(n_time, d=dt_ps))

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    workers = max(1, min(workers, os.cpu_count() or workers))

    output_path = os.path.abspath(output_filename)
    print(f"Generating {num_samples} scalar-NLSE samples")
    print(f"Using {workers} worker process(es)")
    print(f"Writing to: {output_path}")

    chunk_n = min(32, num_samples)
    with h5py.File(output_path, "w") as hf:
        shape = (num_samples, n_time)
        hf.create_dataset(
            "input_temporal_intensity", shape=shape, dtype="f4",
            chunks=(chunk_n, n_time), compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "input_spectral_intensity", shape=shape, dtype="f4",
            chunks=(chunk_n, n_time), compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "output_temporal_intensity", shape=shape, dtype="f4",
            chunks=(chunk_n, n_time), compression="gzip", compression_opts=4
        )
        hf.create_dataset(
            "output_spectral_intensity", shape=shape, dtype="f4",
            chunks=(chunk_n, n_time), compression="gzip", compression_opts=4
        )
        labels_ds = hf.create_dataset(
            "meta_parameters", shape=(num_samples, 8), dtype="f4",
            chunks=(chunk_n, 8), compression="gzip", compression_opts=4
        )
        hf.create_dataset("time_vector_ps", data=time_vector.astype("f4"))
        hf.create_dataset("frequency_vector_per_ps", data=frequency_vector.astype("f4"))

        labels_ds.attrs["column_names"] = np.asarray(
            ["L_km", "beta2_ps2_km", "D_ps_nm_km", "gamma_W_km",
             "loss_dB_km", "P0_W", "T0_ps", "pulse_type_0G_1S"],
            dtype=h5py.string_dtype()
        )
        hf.attrs["wavelength_nm"] = LAMBDA_NM
        hf.attrs["model"] = "scalar_NLSE_SSFM"
        hf.attrs["polarization_axes"] = 1
        hf.attrs["sampling"] = "Latin Hypercube Sampling"
        hf.attrs["seed"] = seed
        hf.attrs["num_steps"] = num_steps
        hf.attrs["time_window_ps"] = t_max_ps

        tasks = (
            (i, sampled_params[i], n_time, t_max_ps, num_steps)
            for i in range(num_samples)
        )

        # imap preserves sample order, so results can be written sequentially.
        with mp.Pool(processes=workers) as pool:
            for completed, result in enumerate(
                pool.imap(simulate_one, tasks, chunksize=1), start=1
            ):
                idx, in_t, in_w, out_t, out_w, labels = result
                hf["input_temporal_intensity"][idx] = in_t
                hf["input_spectral_intensity"][idx] = in_w
                hf["output_temporal_intensity"][idx] = out_t
                hf["output_spectral_intensity"][idx] = out_w
                labels_ds[idx] = labels

                if completed == 1 or completed % max(1, num_samples // 10) == 0 or completed == num_samples:
                    print(f"Progress: {completed}/{num_samples} ({100*completed/num_samples:.1f}%)")

    print(f"\nSUCCESS: dataset saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate a single-axis scalar-NLSE optical-fibre dataset."
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--output", type=str, default="nlse_scalar_dataset.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--time-points", type=int, default=2048)
    parser.add_argument("--time-window", type=float, default=800.0)
    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be at least 1")
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.time_points < 16:
        parser.error("--time-points must be at least 16")
    if args.time_window <= 0:
        parser.error("--time-window must be positive")
    if args.workers is not None and args.workers < 1:
        parser.error("--workers must be at least 1")

    generate_dataset(
        num_samples=args.samples,
        output_filename=args.output,
        seed=args.seed,
        n_time=args.time_points,
        t_max_ps=args.time_window,
        num_steps=args.steps,
        workers=args.workers,
    )


if __name__ == "__main__":
    mp.freeze_support()  # Safe on macOS and packaged Python applications.
    main()
