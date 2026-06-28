"""
Test signal generators for cyclostationary analysis.

Adapted from Fabio Casagrande Hirono's cyclostationarity_analysis
(https://github.com/fchirono/cyclostationarity_analysis, November 2022).
See analysis.py for full attribution.
"""

import numpy as np
import scipy.signal as ss


def create_rect_bpsk(
    T_bits: int,
    num_bits: int,
    fc: float,
    signal_power_dB: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate a rectangular-pulse BPSK signal.

    See https://cyclostationary.blog/2015/09/28/creating-a-simple-cs-signal-rectangular-pulse-bpsk/

    Parameters
    ----------
    T_bits : int
        Samples per bit (1/T_bits is the bit rate in normalized units).
    num_bits : int
        Number of bits in the signal.
    fc : float
        Carrier frequency in normalized units.
    signal_power_dB : float
        Output signal power in dB.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    x_t : np.ndarray
        Complex baseband signal samples, shape (num_bits * T_bits,).
    """
    if rng is None:
        rng = np.random.default_rng()

    N_samples = num_bits * T_bits

    bit_seq = rng.integers(0, 2, num_bits)
    sym_seq = 2 * bit_seq - 1

    zero_mat = np.zeros((T_bits - 1, num_bits))
    sym_seq = np.concatenate((sym_seq[np.newaxis, :], zero_mat), axis=0)
    sym_seq = np.reshape(sym_seq, (N_samples,), order="F")

    p_t = np.ones((T_bits,))
    s_t = ss.lfilter(p_t, [1], sym_seq)

    e_vec = np.exp(1j * 2 * np.pi * fc * np.arange(N_samples))
    x_t = s_t * e_vec

    signal_power = 10 ** (signal_power_dB / 10)
    x_t *= np.sqrt(signal_power / np.var(x_t))

    return x_t


def create_lowpassmod_cos(
    N_samples: int,
    N_filter: int,
    fc_filter: float,
    f_cos: float,
    fs: float,
    signal_power_dB: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate bandlimited Gaussian noise modulating a cosine wave.

    The noise is lowpassed with a Butterworth filter of order N_filter at
    fc_filter Hz, then multiplied by a cosine at f_cos Hz.

    Parameters
    ----------
    N_samples : int
        Number of output samples.
    N_filter : int
        Butterworth filter order for the lowpass noise.
    fc_filter : float
        Lowpass cutoff frequency in Hz.
    f_cos : float
        Cosine carrier frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    signal_power_dB : float
        Output signal power in dB.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    x : np.ndarray
        Real-valued signal samples, shape (N_samples,).
    """
    if rng is None:
        rng = np.random.default_rng()

    butter_sos = ss.butter(N_filter, fc_filter, output="sos", fs=fs)
    xn = rng.normal(loc=0, scale=1, size=N_samples)
    x_lpn = ss.sosfilt(butter_sos, xn)

    t = np.linspace(0, (N_samples - 1) / fs, N_samples)
    xc = np.cos(2 * np.pi * f_cos * t)
    x = x_lpn * xc

    signal_power = 10 ** (signal_power_dB / 10)
    x *= np.sqrt(signal_power / np.var(x))

    return x


def create_noise(
    x_t: np.ndarray,
    SNR_dB: float,
    *,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Generate AWGN that yields a given SNR when added to x_t.

    Noise is real if x_t is real, complex if x_t is complex.

    Parameters
    ----------
    x_t : np.ndarray
        Reference signal used to set noise power.
    SNR_dB : float
        Desired signal-to-noise ratio in dB.
    rng : np.random.Generator or None
        Random number generator for reproducibility.

    Returns
    -------
    noise_t : np.ndarray
        Noise samples with the same shape and dtype class as x_t.
    """
    if rng is None:
        rng = np.random.default_rng()

    N_t = x_t.shape[0]
    signal_power = np.var(x_t)

    noise_t = rng.normal(0, 1, N_t)
    if np.iscomplexobj(x_t):
        noise_t = noise_t + 1j * rng.normal(0, 1, N_t)

    SNR_lin = 10 ** (SNR_dB / 10)
    desired_noise_power = signal_power / SNR_lin
    noise_t *= np.sqrt(desired_noise_power / np.var(noise_t))

    return noise_t
