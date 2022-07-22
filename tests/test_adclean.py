from adclean.cleaner import Cleaner
import pytest
import pandas as pd


@pytest.mark.parametrize(
    "serie, detect_method, replace_method, output",
    [
        (pd.Series([1, 2, 5, 7]), ["standard_deviation"], ["mean"], Cleaner(pd.Series([1, 2, 5, 7]),
                                                                            ["standard_deviation"], ["mean"]))
    ]
)
def test_init(serie, detect_method, replace_method, output):
    assert (Cleaner(serie, detect_method, replace_method).serie == serie)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([1, 2, 6, 3])), [3])
    ]
)
def test_mean(cleaner, output):
    assert (cleaner._mean() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([1, 2, 1000])), [1000])
    ]
)
def test_standard_devation(cleaner, output):
    assert (cleaner._standard_deviation() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([1, 2, 800])), [800])
    ]
)
def test_gaussian_mixture_model(cleaner, output):
    assert (cleaner._gaussian_mixture_model() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([2, 5, 400])), [400])
    ]
)
def test_extreme_value_analysis(cleaner, output):
    assert (cleaner._extreme_value_analysis() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([1, 3, 500])), [500])
    ]
)
def test_local_outlier_factor(cleaner, output):
    assert (cleaner._local_outlier_factor() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([2, 5, 600])), [600])
    ]
)
def test_connectivity_based_outlier_detection(cleaner, output):
    assert (cleaner._connectivity_based_outlier_detection() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([4, 2, 700])), [700])
    ]
)
def test_angular_based_outlier_detection(cleaner, output):
    assert (cleaner._angular_based_outlier_detection() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([2, 6, 450])), [450])
    ]
)
def test_dbscan_clustering(cleaner, output):
    assert (cleaner._dbscan_clustering() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([6, 2, 900])), [900])
    ]
)
def test_kmeans_clustering(cleaner, output):
    assert (cleaner._kmeans_clustering() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([3, 8, 750])), [750])
    ]
)
def test_knearest_neighbor(cleaner, output):
    assert (cleaner._knearest_neighbor() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([3, 4, 650])), [650])
    ]
)
def test_mahalanobis_distance(cleaner, output):
    assert (cleaner._mahalanobis_distance() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([3, 2, 550])), [550])
    ]
)
def test_isolation_forest(cleaner, output):
    assert (cleaner._isolation_forest() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([5, 2, 50])), [50])
    ]
)
def test_robust_random_cut_forest(cleaner, output):
    assert (cleaner._robust_random_cut_forest() == output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Cleaner(pd.Series([5, 6, 150])), [150])
    ]
)
def test_support_vector_machine(cleaner, output):
    assert (cleaner._support_vector_machine() == output)
