import numpy as np
import pytest

from cyclo import CyclostationaryAnalyzer
import cyclo.signals as sig

RNG = np.random.default_rng(0)
FS = 1.0
NDFT = 64
N_ALPHA = 5
ALPHA_VEC = np.linspace(0, FS, N_ALPHA)


def _make_noisy_bpsk(rng=RNG):
    y = sig.create_rect_bpsk(10, 512, 0.05, 0.0, rng=rng)
    n = sig.create_noise(y, 10.0, rng=rng)
    return y + n


def test_output_shapes():
    signal = _make_noisy_bpsk()
    analyzer = CyclostationaryAnalyzer(signal, fs=FS)
    freq, Sxy, cohere = analyzer.cyclic_periodogram(ALPHA_VEC, NDFT)
    assert freq.shape == (NDFT,)
    assert Sxy.shape == (N_ALPHA, NDFT)
    assert cohere.shape == (N_ALPHA, NDFT)


def test_freq_axis_centered():
    signal = _make_noisy_bpsk()
    analyzer = CyclostationaryAnalyzer(signal, fs=FS)
    freq, _, _ = analyzer.cyclic_periodogram(ALPHA_VEC, NDFT)
    assert freq[0] == pytest.approx(-FS / 2)
    assert freq[-1] == pytest.approx(FS / 2 - FS / NDFT)


def test_no_nan_at_alpha_zero():
    # At alpha=0 the principal domain covers all frequencies, so no NaNs expected
    signal = _make_noisy_bpsk()
    analyzer = CyclostationaryAnalyzer(signal, fs=FS)
    freq, Sxy, cohere = analyzer.cyclic_periodogram(np.array([0.0]), NDFT)
    assert not np.any(np.isnan(Sxy))
    assert not np.any(np.isnan(cohere))


def test_principal_domain_masking():
    # At alpha=fs the principal domain collapses to zero width;
    # only the DC bin (|f|=0 is not strictly > 0) survives — everything else is NaN
    signal = _make_noisy_bpsk()
    analyzer = CyclostationaryAnalyzer(signal, fs=FS)
    freq, Sxy, _ = analyzer.cyclic_periodogram(np.array([FS]), NDFT)
    non_dc = freq != 0.0
    assert np.all(np.isnan(Sxy[0, non_dc]))


def test_conjugate_differs_from_non_conjugate():
    signal = _make_noisy_bpsk()
    analyzer = CyclostationaryAnalyzer(signal, fs=FS)
    _, Sxy_nc, _ = analyzer.cyclic_periodogram(ALPHA_VEC, NDFT, mode="non-conj")
    _, Sxy_c, _ = analyzer.cyclic_periodogram(ALPHA_VEC, NDFT, mode="conj")
    assert not np.allclose(np.nansum(np.abs(Sxy_nc)), np.nansum(np.abs(Sxy_c)))


def test_auto_equals_self_cross():
    # Passing y=None should give identical results to passing y=x explicitly
    signal = _make_noisy_bpsk()
    auto = CyclostationaryAnalyzer(signal, fs=FS)
    cross = CyclostationaryAnalyzer(signal, signal, fs=FS)
    _, Sxy_auto, _ = auto.cyclic_periodogram(ALPHA_VEC, NDFT)
    _, Sxy_cross, _ = cross.cyclic_periodogram(ALPHA_VEC, NDFT)
    np.testing.assert_array_equal(Sxy_auto, Sxy_cross)
