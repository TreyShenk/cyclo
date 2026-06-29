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
from scipy import stats

from cyclo.config import CONFIG_PATH, save_config

# Powers of 2 from 2^6 (64) to 2^17 (131072)
_DEFAULT_NDFT_VALUES = [2**k for k in range(6, 18)]


def _pick_best_workers(
    timings_by_worker: dict[int, list[float]],
    alpha: float = 0.05,
    min_speedup: float = 0.05,
) -> tuple[int, str]:
    """
    Starting from workers=1, promote to a higher worker count only when it is
    both statistically significantly faster (two-sample Mann-Whitney U,
    one-sided p < alpha) AND at least min_speedup faster by median.

    Returns the winning worker count and a short reason string.
    """
    # Test order: explicit positive counts ascending, then -1 ("all cores") last
    ordered = sorted(w for w in timings_by_worker if w > 0)
    if -1 in timings_by_worker:
        ordered.append(-1)

    champion = ordered[0]  # baseline: workers=1

    for candidate in ordered[1:]:
        champ_t = timings_by_worker[champion]
        cand_t = timings_by_worker[candidate]

        # H1: candidate times are stochastically less (faster) than champion
        _, p = stats.mannwhitneyu(cand_t, champ_t, alternative="less")
        speedup = np.median(champ_t) / np.median(cand_t) - 1

        if p < alpha and speedup > min_speedup:
            champion = candidate

    # Build a reason string for display
    if champion == ordered[0]:
        reason = "no sig. improvement over w=1"
    else:
        champ_t = timings_by_worker[champion]
        baseline_t = timings_by_worker[ordered[0]]
        total_speedup = np.median(baseline_t) / np.median(champ_t) - 1
        _, p_final = stats.mannwhitneyu(champ_t, baseline_t, alternative="less")
        reason = f"{total_speedup*100:.0f}% faster than w=1 (p={p_final:.3f})"

    return champion, reason


def run_benchmark(
    ndft_values: list[int] | None = None,
    n_alpha: int = 256,
    n_blocks: int = 256,
    n_repeats: int = 25,
    max_ram_gb: float = 4.0,
    alpha: float = 0.05,
    min_speedup: float = 0.05,
    save: bool = True,
) -> dict[int, int]:
    """
    Benchmark scipy.fft workers for each Ndft size and return a mapping of
    Ndft → optimal workers.

    Worker selection uses a sequential Mann-Whitney U test: starting from
    workers=1, a higher count is only chosen if it is statistically
    significantly faster (p < alpha) AND at least min_speedup faster by
    median. This prevents noisy small-FFT measurements from spuriously
    promoting high worker counts where threading adds overhead.

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
        Number of timing trials collected per worker count. All n_repeats
        values are kept for the statistical test (not just the minimum).
    max_ram_gb : float
        Maximum RAM in GB for the pre-allocated block pool. When a single
        Ndft size would exceed this, the pool is capped and blocks are cycled.
    alpha : float
        Significance threshold for the Mann-Whitney U test. Default 0.05.
    min_speedup : float
        Minimum median speedup (as a fraction) required to prefer more workers.
        Default 0.05 (5%). Guards against statistically significant but
        practically irrelevant differences.
    save : bool
        Whether to write results to ~/.cyclo/config.json.

    Returns
    -------
    dict mapping Ndft → optimal workers value
    """
    if ndft_values is None:
        ndft_values = _DEFAULT_NDFT_VALUES

    n_cores = os.cpu_count() or 1
    # Order: positive counts ascending, -1 ("all cores") last
    pos_workers = sorted({1, 2, 4, min(8, n_cores), n_cores})
    worker_options = pos_workers + ([-1] if n_cores not in pos_workers or True else [])
    # Deduplicate while preserving order (n_cores may equal 4 or 8)
    seen: set[int] = set()
    worker_options = [w for w in worker_options if not (w in seen or seen.add(w))]  # type: ignore[func-returns-value]

    max_ram_bytes = int(max_ram_gb * 1024**3)

    print(f"Machine: {n_cores} CPU cores")
    print(f"RAM budget: {max_ram_gb:.1f} GB")
    print(f"Significance: p < {alpha}, min speedup > {min_speedup*100:.0f}%")
    print(f"Testing Ndft: {ndft_values}")
    print(f"Workers tested: {worker_options}")
    print(f"Blocks per trial: {n_blocks}, repeats: {n_repeats}")
    print()

    w_cols = "  ".join(f"w={w:>3}" for w in worker_options)
    header = f"{'Ndft':>8}  {'pool':>6}  {'MB':>6}  {w_cols}  {'winner':<6}  reason"
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

        timings_by_worker: dict[int, list[float]] = {}
        for w in worker_options:
            # warm-up pass
            scipy.fft.fft(blocks[0], axis=-1, workers=w)

            measurements: list[float] = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                for i in range(n_blocks):
                    scipy.fft.fft(blocks[i % pool_size], axis=-1, workers=w)
                measurements.append((time.perf_counter() - t0) * 1000)
            timings_by_worker[w] = measurements

        opt_w, reason = _pick_best_workers(timings_by_worker, alpha, min_speedup)
        best_workers[Ndft] = opt_w

        medians = [np.median(timings_by_worker[w]) for w in worker_options]
        t_cols = "  ".join(f"{m:>7.1f}" for m in medians)
        row = (
            f"{Ndft:>8}  {pool_size:>6}  {pool_ram_mb:>6.1f}  "
            f"{t_cols}  w={opt_w:<4}  {reason}"
        )
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
