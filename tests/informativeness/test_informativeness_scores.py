"""Unit tests for uncertainty sampling methods
"""

import math

import pytest
import torch
from sklearn.cluster import AgglomerativeClustering
from torch.utils.data import DataLoader, TensorDataset

import pyrelational.informativeness.regression as runc
from pyrelational.informativeness import (
    classification_bald,
    classification_entropy,
    classification_least_confidence,
    classification_margin_confidence,
    classification_ratio_confidence,
    softmax,
)
from pyrelational.informativeness import (
    relative_distance,
    representative_sampling,
)


def test_representative_sampling():
    query = torch.randn(100, 50)
    out = representative_sampling(query, num_annotate=5)
    assert len(out) <= 5
    representative_sampling(query, num_annotate=5, clustering_method="AffinityPropagation")

    out = representative_sampling(query, num_annotate=10, clustering_method=AgglomerativeClustering(n_clusters=10))
    assert len(out) <= 10

    dquery = DataLoader(TensorDataset(query), batch_size=1)
    out = representative_sampling(dquery, num_annotate=10)
    assert len(out) <= 10


def test_relative_distance():
    query = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]])
    reference = torch.tensor([[1, 1, 1], [2, 2, 2]])
    dquery = DataLoader(TensorDataset(query), batch_size=1)
    dref = DataLoader(TensorDataset(reference), batch_size=1)

    out = relative_distance(query, reference)
    torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)

    out = relative_distance(dquery, dref)
    torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)

    out = relative_distance(query, dref)
    torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)

    out = relative_distance(dquery, reference)
    torch.testing.assert_close(out, torch.tensor([math.sqrt(3), 0, 0]), rtol=0, atol=1e-3)

    query = torch.randn(100, 10)
    reference = torch.randn(10, 10)
    dquery = DataLoader(TensorDataset(query))
    assert len(relative_distance(dquery, reference)) == 100


def test_regression_uncertainty_tensor_input():
    funcs = [
        "regression_greedy_score",
        "regression_thompson_sampling",
        "regression_least_confidence",
        "regression_bald",
    ]
    a = torch.randn(25, 100)
    for f in funcs:
        fn = getattr(runc, f)
        o = fn(x=a, axis=0)
        assert o.numel() == 100
        o = fn(x=a, axis=1)
        assert o.numel() == 25

    o = runc.regression_expected_improvement(x=a, max_label=0.8, axis=0)
    assert o.numel() == 100
    o = runc.regression_expected_improvement(x=a, max_label=0.8, axis=1)
    assert o.numel() == 25

    o = runc.regression_upper_confidence_bound(x=a, kappa=0.25, axis=0)
    assert o.numel() == 100
    o = runc.regression_upper_confidence_bound(x=a, kappa=0.25, axis=1)
    assert o.numel() == 25

    a = torch.tensor([[0.1, 0.5, 0.5], [0.3, 1, 0.5]])
    o = runc.regression_upper_confidence_bound(x=a, kappa=0.5)
    torch.testing.assert_close(o, torch.tensor([0.2707, 0.9268, 0.5000]), rtol=0, atol=1e-4)


def test_regression_uncertainty_distribution_input():
    funcs = ["regression_greedy_score", "regression_least_confidence"]
    a = torch.distributions.Normal(torch.randn(10), torch.abs(torch.randn(10)))
    for f in funcs:
        fn = getattr(runc, f)
        o = fn(x=a)
        assert o.numel() == 10

    o = runc.regression_expected_improvement(x=a, max_label=0.8)
    assert o.numel() == 10

    o = runc.regression_upper_confidence_bound(x=a, kappa=0.25)
    assert o.numel() == 10


def test_regression_uncertainty_mean_std_input():
    mean, std = torch.randn(10), torch.abs(torch.randn(10))
    o = runc.regression_greedy_score(mean=mean)
    assert o.numel() == 10

    o = runc.regression_least_confidence(std=std)
    assert o.numel() == 10

    o = runc.regression_expected_improvement(mean=mean, std=std, max_label=0.8)
    assert o.numel() == 10

    o = runc.regression_upper_confidence_bound(mean=mean, std=std, kappa=0.25)
    assert o.numel() == 10


def test_regression_input_check():

    with pytest.raises(ValueError) as err:
        runc._check_regression_informativeness_input()
        assert str(err.value) == "Not all of x, mean, and std can be None."

    a = torch.distributions.Normal(torch.randn(10), torch.abs(torch.randn(10)))
    x, m, s = runc._check_regression_informativeness_input(a)
    assert x is None
    assert m.numel() == 10 and s.numel() == 10

    a = torch.distributions.Normal(torch.randn(10, 2), torch.abs(torch.randn(10, 2)))
    with pytest.raises(AssertionError) as err:
        runc._check_regression_informativeness_input(a)
        assert str(err.value) == "distribution input should be 1D"

    a = torch.randn(25, 100, 1)
    x, m, s = runc._check_regression_informativeness_input(a)
    assert x.ndim == 2
    assert m.ndim == 1 and s.ndim == 1
    assert m.numel() == 100 and s.numel() == 100

    mean, std = torch.randn(10), torch.abs(torch.randn(10))
    x, m, s = runc._check_regression_informativeness_input(mean=mean, std=std)
    assert x is None
    assert m.numel() == 10 and s.numel() == 10


def test_least_confidence():
    prob_dist = torch.tensor([0.1, 0.5, 0.4])
    assert classification_least_confidence(prob_dist) == pytest.approx(0.75, 0.01)
    prob_dist = torch.tensor([[0.5, 0.4, 0.1], [0.5, 0.4, 0.1]])
    out = classification_least_confidence(prob_dist)
    assert out == pytest.approx(0.75, 0.01)
    assert out.numel() == 2


def test_margin_confidence():
    prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
    assert classification_margin_confidence(prob_dist) == pytest.approx(0.9, 0.01)


def test_classificaiton_bald():
    prob_dist = torch.tensor([[0.1, 0.5, 0.4], [0.3, 0.3, 0.4]]).unsqueeze(1)
    assert classification_bald(prob_dist) == pytest.approx(0.035, 0.01)


def test_ratio_confidence():
    prob_dist = torch.tensor([[0.1, 0.5, 0.4]])
    assert classification_ratio_confidence(prob_dist) == pytest.approx(0.8, 0.01)


def test_entropy_based():
    prob_dist = torch.tensor([0.1, 0.5, 0.4])
    assert classification_entropy(prob_dist) == pytest.approx(0.8587, 0.01)


def test_softmax():
    scores = torch.tensor([1.0, 4.0, 2.0, 3.0])
    answer = torch.tensor([0.0321, 0.6439, 0.0871, 0.2369])
    assert torch.allclose(softmax(scores, base=math.e), answer, rtol=1e-3, atol=1e-4)
    assert sum(softmax(scores, base=math.e)).item() == pytest.approx(1, 0.1)
