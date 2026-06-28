"""
Cyclostationary signal analysis using the Time-Smoothing Method (TSM).

Adapted from Fabio Casagrande Hirono's cyclostationarity_analysis
(https://github.com/fchirono/cyclostationarity_analysis, November 2022),
which draws from Chad Spooner's Cyclostationary Signal Processing blog
(https://cyclostationary.blog/) and:

    W. A. Gardner, "Exploitation of Spectral Redundancy in Cyclostationary
    Signals," IEEE Signal Processing Magazine, vol. 8, no. 2, pp. 14-36, 1991.
"""

import numpy as np
import scipy.fft

from cyclo.config import get_workers


class CyclostationaryAnalyzer:
    """
    Cyclostationary signal analyzer using the Time-Smoothing Method.

    Parameters
    ----------
    x : np.ndarray
        1-D array of time-domain signal samples.
    y : np.ndarray or None
        Second signal for cross-spectral analysis. If None, auto-correlation
        is performed (y = x).
    fs : float
        Sampling frequency in Hz. Default is 1.0.
    """

    def __init__(
        self,
        x: np.ndarray,
        y: np.ndarray | None = None,
        fs: float = 1.0,
    ) -> None:
        self.x = np.asarray(x)
        self.y = np.asarray(y) if y is not None else self.x
        self.fs = float(fs)

    def cyclic_periodogram(
        self,
        alpha_vec: np.ndarray,
        Ndft: int,
        mode: str = "non-conj",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cyclic spectral density and cyclic spectral coherence using
        the periodogram (time-smoothing) method.

        Parameters
        ----------
        alpha_vec : np.ndarray
            Array of cyclic frequencies in Hz, shape (N_alpha,).
        Ndft : int
            Number of DFT points per block.
        mode : {'non-conj', 'conj'}
            Whether to compute non-conjugate or conjugate spectral correlation.

        Returns
        -------
        freq : np.ndarray
            Frequency axis in Hz, shape (Ndft,), centered at 0.
        Sxy_avg : np.ndarray
            Cyclic spectral density, shape (N_alpha, Ndft). Complex.
            Entries outside the principal domain are NaN.
        cohere : np.ndarray
            Cyclic spectral coherence, shape (N_alpha, Ndft). Complex.
            Entries outside the principal domain are NaN.
        """
        alpha_vec = np.asarray(alpha_vec)
        N_alpha = alpha_vec.shape[0]
        fs = self.fs

        Nt = self.x.shape[0]
        N_blocks = Nt // Ndft

        df = fs / Ndft
        freq = np.linspace(0, fs - df, Ndft) - fs / 2

        Sxx = np.zeros((N_blocks, N_alpha, Ndft))
        Syy = np.zeros((N_blocks, N_alpha, Ndft))
        Sxy = np.zeros((N_blocks, N_alpha, Ndft), dtype="complex128")

        y = self.y.conj() if mode == "conj" else self.y
        workers = get_workers(Ndft)

        for n in range(N_blocks):
            n_start = n * Ndft
            t_block = np.linspace(n_start / fs, (n_start + Ndft) / fs, Ndft)
            x_block = self.x[n_start : n_start + Ndft]
            y_block = y[n_start : n_start + Ndft]

            # Vectorize over all alphas at once: shape (N_alpha, Ndft)
            shifts = np.exp(-1j * np.pi * alpha_vec[:, np.newaxis] * t_block[np.newaxis, :])
            u_blocks = x_block[np.newaxis, :] * shifts
            v_blocks = y_block[np.newaxis, :] * shifts.conj()

            u_f = scipy.fft.fft(u_blocks, axis=-1, workers=workers)
            v_f = scipy.fft.fft(v_blocks, axis=-1, workers=workers)

            Sxx[n] = (u_f * u_f.conj()).real
            Syy[n] = (v_f * v_f.conj()).real
            Sxy[n] = u_f * v_f.conj()

        scale = 1.0 / (Ndft * fs)
        Sxx *= scale
        Syy *= scale
        Sxy *= scale

        Sxx = np.fft.fftshift(Sxx, axes=-1)
        Syy = np.fft.fftshift(Syy, axes=-1)
        Sxy = np.fft.fftshift(Sxy, axes=-1)

        Sxx_avg = Sxx.mean(axis=0)
        Syy_avg = Syy.mean(axis=0)
        Sxy_avg = Sxy.mean(axis=0)

        cohere = Sxy_avg / np.sqrt(Sxx_avg * Syy_avg)

        for a, alpha in enumerate(alpha_vec):
            outside = np.abs(freq) > (fs - np.abs(alpha)) / 2
            Sxy_avg[a, outside] = np.nan
            cohere[a, outside] = np.nan

        return freq, Sxy_avg, cohere

    @staticmethod
    def _calc_xspec_block(
        x: np.ndarray,
        y: np.ndarray,
        t: np.ndarray,
        alpha: float = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute cross-spectrum for one data block with cyclic frequency shift.

        Parameters
        ----------
        x : np.ndarray
            Block of N samples from signal x.
        y : np.ndarray
            Block of N samples from signal y.
        t : np.ndarray
            Time values for the block in seconds, shape (N,).
        alpha : float
            Cyclic frequency in Hz.

        Returns
        -------
        Suu : np.ndarray
            Auto-power spectrum of x shifted by +alpha/2.
        Svv : np.ndarray
            Auto-power spectrum of y shifted by -alpha/2.
        Suv : np.ndarray
            Cross-power spectrum between shifted x and y.
        """
        u = x * np.exp(-1j * np.pi * alpha * t)
        v = y * np.exp(+1j * np.pi * alpha * t)

        u_f = np.fft.fft(u)
        v_f = np.fft.fft(v)

        Suu = (u_f * u_f.conj()).real
        Svv = (v_f * v_f.conj()).real
        Suv = u_f * v_f.conj()

        return Suu, Svv, Suv
