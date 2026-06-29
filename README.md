# cyclo

[![CI](https://github.com/TreyShenk/cyclo/actions/workflows/ci.yml/badge.svg)](https://github.com/TreyShenk/cyclo/actions/workflows/ci.yml)

Python library for cyclostationary signal analysis, built around the **Time-Smoothing Method (TSM)** for computing cyclic spectral density and cyclic spectral coherence.

## Background

A signal is *cyclostationary* if its statistical properties vary periodically with time. Many real-world signals exhibit this — BPSK, AM, and other digitally modulated waveforms all have spectral redundancy at specific *cyclic frequencies* (α) that can be exploited for detection and classification, even in noise.

This library computes the **Spectral Correlation Density (SCD)** and **Spectral Coherence** over a user-specified range of cyclic frequencies using the time-smoothing periodogram method described in:

> W. A. Gardner, "Exploitation of Spectral Redundancy in Cyclostationary Signals," *IEEE Signal Processing Magazine*, vol. 8, no. 2, pp. 14–36, 1991.

## Attribution

The algorithms here are adapted from [Fabio Casagrande Hirono's cyclostationarity_analysis](https://github.com/fchirono/cyclostationarity_analysis) (November 2022), which itself draws from [Chad Spooner's Cyclostationary Signal Processing blog](https://cyclostationary.blog/) — the primary reference for this field. No algorithms were changed in the refactor; only structure and performance were updated.

## Authorship note

This library was written by [Claude](https://claude.ai/code) (Anthropic's AI coding assistant), guided by a DSP engineer. The author has solid signal processing fundamentals but is **not** a cyclostationary signal processing expert. If you find an error in the implementation or documentation — whether algorithmic, terminological, or conceptual — please open an issue. Corrections are genuinely welcome.

## Installation

Requires Python ≥ 3.10 and [uv](https://docs.astral.sh/uv/).

**As a dependency in your own project** — add it directly from GitHub:

```bash
uv add git+https://github.com/TreyShenk/cyclo.git
```

Or with pip:

```bash
pip install git+https://github.com/TreyShenk/cyclo.git
```

**To work on the package itself** — clone and install in editable mode:

```bash
git clone https://github.com/TreyShenk/cyclo.git
cd cyclo
uv sync
```

`uv sync` installs the package in editable mode along with all dependencies, so changes to `src/cyclo/` take effect immediately without reinstalling. For the demo notebook:

```bash
uv sync --extra notebook
uv run jupyter notebook notebooks/demo.ipynb
```

## Usage

```python
import numpy as np
from cyclo import CyclostationaryAnalyzer
import cyclo.signals as sig

fs = 1.0
rng = np.random.default_rng(42)

# Generate a rectangular-pulse BPSK signal and add noise
y = sig.create_rect_bpsk(T_bits=10, num_bits=32768, fc=0.05, signal_power_dB=0.0, rng=rng)
n = sig.create_noise(y, SNR_dB=10.0, rng=rng)

# Compute non-conjugate cyclic spectral density and coherence
analyzer = CyclostationaryAnalyzer(y + n, fs=fs)
alpha_vec = np.linspace(0, fs, 21)
freq, Sxy, cohere = analyzer.cyclic_periodogram(alpha_vec, Ndft=256)

# freq:   frequency axis, shape (Ndft,)
# Sxy:    cyclic spectral density, shape (N_alpha, Ndft) — NaN outside principal domain
# cohere: cyclic spectral coherence, shape (N_alpha, Ndft)
```

Pass `mode='conj'` to compute the conjugate spectral correlation (useful for locating features near 2·f_c in BPSK-type signals):

```python
freq, Sxy_c, cohere_c = analyzer.cyclic_periodogram(alpha_vec, Ndft=256, mode='conj')
```

For cross-spectral analysis between two signals, pass both to the constructor:

```python
analyzer = CyclostationaryAnalyzer(x, y, fs=fs)
```

## Development

Dev dependencies (nox, pytest) are in a separate group and not installed for regular users:

```bash
uv sync --group dev
```

Run the test suite across all configured Python versions:

```bash
uv run nox
```

Or run pytest directly in the current environment:

```bash
uv run pytest tests/
```

To run a single test file or test by name:

```bash
uv run pytest tests/test_basic.py::test_output_shapes
```

**Python version coverage:** tested on 3.10–3.14 via nox and CI. On 3.10/3.11 the resolver picks numpy 2.2.x / scipy 1.15.x; on 3.12+ it picks the latest (numpy 2.5.x / scipy 1.18.x). Add versions to `PYTHON_VERSIONS` in `noxfile.py` as new Python releases arrive.

## Performance tuning

Worker count for `scipy.fft` is automatically read from `~/.cyclo/config.json` if present. Run the benchmark once to generate this file tuned to your machine:

```bash
uv run cyclo-benchmark
```

The benchmark tests powers of 2 from 64 to 131072 and respects a configurable RAM budget (default 4 GB) to avoid swap at large FFT sizes. Results are picked up automatically on all subsequent runs — no code changes needed.
