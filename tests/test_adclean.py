import pandas as pd
from adclean.cleaner import Cleaner
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
