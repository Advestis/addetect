from adclean.cleaner import Cleaner
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "serie, detect_method, replace_method, output",
    [
        (pd.Series([1, 2, 3, 4]), None, None, pd.Series([1, 2, 3, 4])),
        (pd.Series([1, 2, 3, 4]), ["test1"], None,
         f"The detection method test1 does not exist in {['standard_deviation', 'gaussian_mixture_model', 'extreme_value_analysis', 'local_outlier_factor', 'connectivity_based_outlier_detection', 'angular_based_outlier_detection', 'dbscan_clustering', 'kmeans_clustering', 'knearest_neighbor', 'mahalanobis_distance', 'isolation_forest', 'support_vector_machine']}"),
        (pd.Series([1, 2, 3, 4]), None, ["test1"],
         f"The replace method test1 does not exist in {['median']}"),
        ([1, 2, 7, 1000], None, None, f"The type of the series must be pd.Series and not {type([1, 2, 7, 1000])}")
    ]
)
def test_init(serie, detect_method, replace_method, output):
    try:
        assert Cleaner(serie, detect_method, replace_method).serie.equals(output)
    except Exception as e:
        assert output in str(e)


@pytest.mark.parametrize(
    "cleaner, maximum, output",
    [
        (Cleaner(serie=pd.Series([200, 210, 700, 240]), detect_method=["maximum"]), 500, [700])
    ]
)
def test_max(cleaner, maximum, output):
    assert cleaner._maximum(maximum) == output


@pytest.mark.parametrize(
    "cleaner, minimum, output",
    [
        (Cleaner(serie=pd.Series([200, 210, 5, 240]), detect_method=["minimum"]), 100, [5])
    ]
)
def test_min(cleaner, minimum, output):
    assert cleaner._minimum(minimum) == output


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