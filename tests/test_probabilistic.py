import numpy as np
import pytest
import torch

from fd_airquality.probabilistic import quantile_statistics, rearrange_around_median, sample_crps


def test_quantile_metrics_mask_truth_but_never_invalid_forecasts():
    truth = np.array([[1., np.nan, 3.]])
    quantiles = np.array([[[0., 1., 2.], [0., 1., 2.], [2., 3., 4.]]])
    stats = quantile_statistics(truth, quantiles, [.1, .5, .9])
    assert stats["quantile-count"] == 6
    assert stats["quantile-loss-sum"] == pytest.approx(.8)
    assert stats["wis-sum"] == pytest.approx(.4 / 1.5)
    assert stats["interval-80-covered"] == 2
    assert stats["interval-80-width-sum"] == 4
    assert stats["quantile-crossings"] == 0
    assert stats["crps-count"] == 2
    assert stats["crps-sum"] == pytest.approx(.2)
    quantiles[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="Cuantiles"):
        quantile_statistics(truth, quantiles, [.1, .5, .9])


def test_crossing_quantiles_are_reported_not_silently_sorted():
    stats = quantile_statistics(np.array([[1.]]), np.array([[[3., 1., 0.]]]), [.1, .5, .9])
    assert stats["quantile-crossings"] == 1


def test_median_anchored_quantiles_are_monotone_without_changing_the_point():
    raw = torch.tensor([[[4., 1., 2., 6., 0.], [0., 1., 2., 3., 4.]]])
    expected = torch.tensor([[[1., 2., 2., 2., 6.], [0., 1., 2., 3., 4.]]])
    fixed = rearrange_around_median(raw, 2)
    torch.testing.assert_close(fixed, expected, atol=0, rtol=0)
    torch.testing.assert_close(fixed[..., 2], raw[..., 2], atol=0, rtol=0)
    torch.testing.assert_close(rearrange_around_median(fixed, 2), fixed, atol=0, rtol=0)


def test_crossings_are_audited_on_original_and_published_quantiles():
    truth = np.array([[1., np.nan]])
    raw = np.array([[[3., 1., 0.], [8., 1., 0.]]])
    fixed = np.ones_like(raw)
    stats = quantile_statistics(truth, fixed, [.1, .5, .9], raw_quantiles=raw)
    assert stats["quantile-raw-crossings"] == stats["probabilistic-targets"] == 1
    assert stats["quantile-crossings"] == stats["interval-80-width-sum"] == stats["crps"] == 0
    assert stats["interval-80-covered"] == 1
    raw[0, 1, 0] = np.inf
    with pytest.raises(ValueError, match="Cuantiles originales"):
        quantile_statistics(truth, fixed, [.1, .5, .9], raw_quantiles=raw)


def test_empirical_crps_matches_independent_pairwise_definition():
    truth = np.array([[2., np.nan], [3., 4.]])
    samples = np.array([[[1., 2., 5.], [0., 1., 2.]], [[1., 4., 9.], [4., 4., 4.]]])
    expected = []
    for y, draws in zip(truth[np.isfinite(truth)], samples[np.isfinite(truth)], strict=True):
        expected.append(np.abs(draws - y).mean() - np.abs(draws[:, None] - draws[None, :]).mean() / 2)
    np.testing.assert_allclose(sample_crps(truth, samples), expected)
    assert sample_crps(np.array([3.]), np.array([[5.]]))[0] == 2.
    samples[0, 1, 0] = np.nan
    with pytest.raises(ValueError, match="CRPS"):
        sample_crps(truth, samples)


def test_quantile_crps_has_full_unit_interval_weights_even_on_an_uneven_grid():
    truth = np.array([[2., np.nan, 7.]])
    quantiles = np.array([[[5., 5., 5.], [99., 99., 99.], [5., 5., 5.]]])
    stats = quantile_statistics(truth, quantiles, [.01, .5, .8])
    # A degenerate distribution has CRPS equal to its absolute point error.
    assert stats["crps"] == pytest.approx((3 + 2) / 2)
    assert stats["crps-count"] == 2
