# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies and the local package
uv sync

# Install with notebook extras
uv sync --extra notebook

# Run a script / one-off check
uv run python -c "..."

# Run a specific test file (no test framework yet; add pytest when tests are introduced)
uv run python path/to/test.py
```

## Architecture

This is a Python library for cyclostationary signal analysis, adapted from Fabio Casagrande Hirono's [cyclostationarity_analysis](https://github.com/fchirono/cyclostationarity_analysis).

```
src/cyclo/
  __init__.py     # exports CyclostationaryAnalyzer and signals submodule
  analysis.py     # CyclostationaryAnalyzer class — core analysis logic
  signals.py      # test signal generators (BPSK, lowpass-mod cosine, AWGN)
notebooks/
  demo.ipynb      # end-to-end demonstration with all plots
```

**`CyclostationaryAnalyzer`** (`analysis.py`) is the main entry point. Instantiate with a signal `x` (and optionally a second signal `y` for cross-correlation) plus sampling frequency `fs`. Call `cyclic_periodogram(alpha_vec, Ndft, mode)` to get `(freq, Sxy_avg, cohere)`.

The algorithm is the **Time-Smoothing Method**: the signal is divided into `N_blocks = len(x) // Ndft` non-overlapping blocks; for each block and each cyclic frequency `alpha`, `_calc_xspec_block` applies ±alpha/2 frequency shifts before computing FFTs and cross/auto spectra; results are averaged across blocks; entries outside the principal domain (`|f| > (fs - |alpha|)/2`) are set to NaN.

`mode='conj'` computes the conjugate spectral correlation (conjugates `y` before processing), which reveals features at cyclic frequencies near `2*fc` for BPSK-type signals.

**`signals.py`** is for test signal generation only — not part of the analysis API. All three generators accept an optional `rng: np.random.Generator` keyword argument for reproducibility.

## Attribution

All algorithm implementations credit: Fabio Casagrande Hirono, Chad Spooner's Cyclostationary Signal Processing blog, and W. A. Gardner (IEEE SPM, 1991). Do not change the algorithms without noting the deviation from the reference.
