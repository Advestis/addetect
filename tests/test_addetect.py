import numpy as np

from addetect.detector import Detector
import pandas as pd
import pytest


@pytest.mark.parametrize(
    "serie, detect_method, replace_method, output",
    [
        (pd.Series([1, 2, 3, 4]), None, None, pd.Series([1, 2, 3, 4])),
        (pd.Series([1, 2, 3, 4]), ["test1"], None, ValueError("The detection method test1 does not exist")),
        (pd.Series([1, 2, 3, 4]), None, ["test1"], ValueError("he replace method test1 does not exist")),
        ([1, 2, 7, 1000], None, None, TypeError("The type of the series must be pd.Series"))
    ]
)
def test_init(serie, detect_method, replace_method, output):
    if isinstance(output, Exception):
        with pytest.raises(output.__class__) as e:
            Detector(serie, detect_method, replace_method)
        assert str(output) in str(e.value)
    else:
        assert Detector(serie, detect_method, replace_method).serie.equals(output)


# @pytest.mark.parametrize(
#     "cleaner, maximum, output",
#     [
#         (Detector(serie=pd.Series([200, 210, 700, 240]), detect_methods=["maximum"]), 500, [700])
#     ]
# )
# def test_max(cleaner, maximum, output):
#     assert cleaner._maximum(maximum) == output
#
#
# @pytest.mark.parametrize(
#     "cleaner, minimum, output",
#     [
#         (Detector(serie=pd.Series([200, 210, 5, 240]), detect_methods=["minimum"]), 100, [5])
#     ]
# )
# def test_min(cleaner, minimum, output):
#     assert cleaner._minimum(minimum) == output


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Detector(serie=pd.Series([10, 12, 14, 15, 16, 19, 20, 21, 22, 159, 180],
                                  index=pd.date_range(start="2022-01-01", end="2022-01-11"), name="value")),

         pd.Series([159, 180], index=pd.date_range(start="2022-01-10", end="2022-01-11"), name="value")),

        (Detector(
            serie=pd.Series([10, 12, 14, 15, 16],
                            index=["2022-01-01", "2022-01-02", "2022-01-03", "2022-01-04", "2022-01-05"],
                            name="value")),
         pd.Series([], index=[], name="value", dtype=int))
    ]
)
def test_iqr(cleaner, output):
    res = cleaner._iqr()
    assert res.equals(output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")), [0.6283741665788453,
                                                                                        0.5788958857458654,
                                                                                        0.5294176049128854,
                                                                                        0.5046784644963954,
                                                                                        0.4799393240799054,
                                                                                        0.4057219028304355,
                                                                                        0.3809827624139455,
                                                                                        0.3562436219974556,
                                                                                        0.33150448158096557,
                                                                                        0.3067653411644756,
                                                                                        0.2820262007479856,
                                                                                        0.2572870603314957,
                                                                                        0.2325479199150057,
                                                                                        0.20780877949851573,
                                                                                        0.18306963908202575,
                                                                                        0.15833049866553578,
                                                                                        0.1335913582490458,
                                                                                        0.3562436219974556,
                                                                                        0.2820262007479856,
                                                                                        0.009895656166595953,
                                                                                        0.08906090549936393,
                                                                                        0.3562436219974556,
                                                                                        0.23749574799830378,
                                                                                        3.0577577554781605,
                                                                                        3.5772797042244497, ])
    ]
)
def test_get_z_score(cleaner, output):
    j = 0
    for i in cleaner.serie.values:
        res = cleaner.get_zscore(i)
        assert res == output[j]
        j += 1


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")),
         pd.Series([159, 180], index=pd.date_range(start="2022-01-24", end="2022-01-25"), name="value"))
    ]
)
def test_z_score(cleaner, output):
    assert cleaner._zscore().equals(output)


@pytest.mark.parametrize(
    "cleaner, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")),
         pd.Series([159, 180], index=pd.date_range(start="2022-01-24", end="2022-01-25"), name="value")),

        (Detector(pd.Series([0.047, 0.83, 0.91, 1.03, 0.62], index=pd.date_range(start="2022-02-12", end="2022-02-16"),
                            name="value")), ValueError("The series does not follow a normal law, so we cannot use this "
                                                      "method")
         )
    ]
)
def test_standard_deviation(cleaner, output):
    if isinstance(output, Exception):
        with pytest.raises(output.__class__) as e:
            cleaner._standard_deviation()
        assert str(output) in str(e.value)
    else:
        assert cleaner._standard_deviation().equals(output)


@pytest.mark.parametrize(
    "cleaner, alpha_level, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")), 0.5, True),

        (Detector(pd.Series([0.047, 0.83, 0.91, 1.03, 0.62], index=pd.date_range(start="2022-02-12", end="2022-02-16"),
                            name="value")), 0.05, False)
    ]
)
def test_verif_norm(cleaner, alpha_level, output):
    assert cleaner._verif_norm() == output


@pytest.mark.parametrize(
    "cleaner, outliers, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")),
         pd.Series([159, 180], index=pd.date_range(start="2022-01-24", end="2022-01-25"), name="value"),

         Detector(serie=pd.Series(
             [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 24, 24],
             index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")).serie
        )
    ]
)
def test_replace_by_median(cleaner, outliers, output):
    assert cleaner.replace_by_median(outliers).equals(output)


@pytest.mark.parametrize(
    "cleaner, outliers, output",
    [
        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")),
         pd.Series([159, 180], index=pd.date_range(start="2022-01-24", end="2022-01-25"), name="value"),

         Detector(serie=pd.Series(
             [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 45, 45],
             index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")).serie),

        (Detector(serie=pd.Series(
            [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 159, 24, 35, 39, 21, 45, 47, 180],
            index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")),
         pd.Series([159, 180], index=[pd.to_datetime("2022-01-18"),
                                      pd.to_datetime("2022-01-25")], name="value"),

         Detector(serie=pd.Series(
             [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 30, 24, 35, 39, 21, 45, 47, 47],
             index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value")).serie
        ),
    ]
)
def test_replace_by_post_value(cleaner, outliers, output):
    assert cleaner.replace_by_post_value(outliers).equals(output)
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([1, 2, 1000])), [1000])
#     ]
# )
# def test_standard_devation(cleaner, output):
#     assert (cleaner._standard_deviation() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([1, 2, 800])), [800])
#     ]
# )
# def test_gaussian_mixture_model(cleaner, output):
#     assert (cleaner._gaussian_mixture_model() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([2, 5, 400])), [400])
#     ]
# )
# def test_extreme_value_analysis(cleaner, output):
#     assert (cleaner._extreme_value_analysis() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([1, 3, 500])), [500])
#     ]
# )
# def test_local_outlier_factor(cleaner, output):
#     assert (cleaner._local_outlier_factor() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([2, 5, 600])), [600])
#     ]
# )
# def test_connectivity_based_outlier_detection(cleaner, output):
#     assert (cleaner._connectivity_based_outlier_detection() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([4, 2, 700])), [700])
#     ]
# )
# def test_angular_based_outlier_detection(cleaner, output):
#     assert (cleaner._angular_based_outlier_detection() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([2, 6, 450])), [450])
#     ]
# )
# def test_dbscan_clustering(cleaner, output):
#     assert (cleaner._dbscan_clustering() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([6, 2, 900])), [900])
#     ]
# )
# def test_kmeans_clustering(cleaner, output):
#     assert (cleaner._kmeans_clustering() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([3, 8, 750])), [750])
#     ]
# )
# def test_knearest_neighbor(cleaner, output):
#     assert (cleaner._knearest_neighbor() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([3, 4, 650])), [650])
#     ]
# )
# def test_mahalanobis_distance(cleaner, output):
#     assert (cleaner._mahalanobis_distance() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([3, 2, 550])), [550])
#     ]
# )
# def test_isolation_forest(cleaner, output):
#     assert (cleaner._isolation_forest() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([5, 2, 50])), [50])
#     ]
# )
# def test_robust_random_cut_forest(cleaner, output):
#     assert (cleaner._robust_random_cut_forest() == output)
#
#
# @pytest.mark.parametrize(
#     "cleaner, output",
#     [
#         (Detector(pd.Series([5, 6, 150])), [150])
#     ]
# )
# def test_support_vector_machine(cleaner, output):
#     assert (cleaner._support_vector_machine() == output)
