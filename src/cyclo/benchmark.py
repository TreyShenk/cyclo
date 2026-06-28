"""
Benchmark scipy.fft worker counts across Ndft sizes to find the fastest
configuration for this machine. Results are saved to ~/.cyclo/config.json
and used automatically by CyclostationaryAnalyzer.

Run from the command line:

    uv run cyclo-benchmark
"""

import os
import time
from datetime import datetime, timezone

import numpy as np
import scipy.fft

from cyclo.config import CONFIG_PATH, save_config

# Powers of 2 from 2^6 (64) to 2^17 (131072)
_DEFAULT_NDFT_VALUES = [2**k for k in range(6, 18)]


def run_benchmark(
    ndft_values: list[int] | None = None,
    n_alpha: int = 256,
    n_blocks: int = 256,
    n_repeats: int = 25,
    max_ram_gb: float = 4.0,
    save: bool = True,
) -> dict[int, int]:
    """
    Benchmark scipy.fft workers for each Ndft size and return a mapping of
    Ndft → optimal workers.

    Parameters
    ----------
    ndft_values : list[int] or None
        FFT sizes to test. Defaults to powers of 2 from 64 to 131072 (2^6–2^17).
    n_alpha : int
        Number of alpha values (sets the batch row count for each FFT call).
        Should match the N_alpha you typically pass to cyclic_periodogram.
    n_blocks : int
        Number of FFT calls per timing trial. More = more stable, but slower
        at large Ndft. Reduce to 32–64 if the benchmark takes too long.
    n_repeats : int
        Number of timing trials; the minimum time is kept to reduce noise.
    max_ram_gb : float
        Maximum RAM in GB to use for pre-allocated block pool. When a single
        Ndft size would exceed this, the pool is capped and blocks are cycled
        during timing. Default is 1.0 GB.
    save : bool
        Whether to write results to ~/.cyclo/config.json.

    Returns
    -------
    dict mapping Ndft → optimal workers value
    """
    if ndft_values is None:
        ndft_values = _DEFAULT_NDFT_VALUES

    n_cores = os.cpu_count() or 1
    worker_options = sorted({1, 2, 4, min(8, n_cores), n_cores, -1})
    max_ram_bytes = int(max_ram_gb * 1024**3)

    print(f"Machine: {n_cores} CPU cores")
    print(f"RAM budget: {max_ram_gb:.1f} GB")
    print(f"Testing Ndft: {ndft_values}")
    print(f"Testing workers: {worker_options}")
    print(f"Blocks per trial: {n_blocks}, repeats: {n_repeats}")
    print()

    w_cols = "  ".join(f"w={w:>3}" for w in worker_options)
    header = f"{'Ndft':>8}  {'pool':>6}  {'MB':>6}  {w_cols}"
    print(header)
    print("-" * len(header))

    best_workers: dict[int, int] = {}

    for Ndft in ndft_values:
        bytes_per_block = n_alpha * Ndft * 16  # complex128 = 16 bytes
        pool_size = max(1, min(n_blocks, max_ram_bytes // bytes_per_block))
        pool_ram_mb = pool_size * bytes_per_block / 1024**2

        blocks = [
            np.random.randn(n_alpha, Ndft) + 1j * np.random.randn(n_alpha, Ndft)
            for _ in range(pool_size)
        ]

        timings: list[float] = []
        for w in worker_options:
            # warm-up pass
            scipy.fft.fft(blocks[0], axis=-1, workers=w)

            best = float("inf")
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for i in range(n_blocks):
                    scipy.fft.fft(blocks[i % pool_size], axis=-1, workers=w)
                elapsed = time.perf_counter() - t0
                best = min(best, elapsed)
            timings.append(best * 1000)

        opt_idx = timings.index(min(timings))
        opt_w = worker_options[opt_idx]
        best_workers[Ndft] = opt_w

        t_cols = "  ".join(f"{t:>7.1f}" for t in timings)
        row = f"{Ndft:>8}  {pool_size:>6}  {pool_ram_mb:>6.1f}  {t_cols}  <- w={opt_w}"
        print(row)

    print()
    print("Optimal workers per Ndft:")
    for Ndft, w in best_workers.items():
        print(f"  Ndft={Ndft}: workers={w}")

    if save:
        config = {
            "generated": datetime.now(timezone.utc).isoformat(),
            "cpu_count": n_cores,
            "workers_by_ndft": {str(k): v for k, v in best_workers.items()},
        }
        save_config(config)
        print(f"\nConfig saved to {CONFIG_PATH}")

    return best_workers


if __name__ == "__main__":
    run_benchmark()
