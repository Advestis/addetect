from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class Cleaner:
    DETEC_METHOD = [
        "standard_deviation",
        "iqr",
        "zscore",
        "gaussian_mixture_model",
        "extreme_value_analysis",
        "local_outlier_factor",
        "connectivity_based_outlier_detection",
        "angular_based_outlier_detection",
        "dbscan_clustering",
        "kmeans_clustering",
        "knearest_neighbor",
        "mahalanobis_distance",
        "isolation_forest",
        "support_vector_machine",
        "minimum",
        "maximum",
    ]

    # TODO : Add the different replace methods
    REPL_METHOD = ["median"]

    # TODO : Verif the type of the index @eguir HINT (pcotte): use 'adtypingdecorator'
    def __init__(
            self,
            serie: Optional[pd.Series] = None,
            detect_methods: Optional[List[str]] = None,
            replace_method: Optional[List[str]] = None,
    ):
        """
        The constructor of the Cleaner class

        Parameters
        ----------
        serie : pd.Series
            If None, will have to be specified later by setting the instance's 'series' attribute
        detect_methods : List[str]
            If None, will default to 'standard_deviation'
        replace_method : List[str]
            If None, will defaul to 'median'

        Raises
        ---------
        TypeError
            If 'series' is not a pd.Series object
        ValueError
            If any of the detection methods or replace methods is unknown by the class
        """
        if not isinstance(serie, pd.Series):
            raise TypeError(f"The type of the series must be pd.Series and not {type(serie)}")
        self._serie = serie
        if detect_methods is not None:
            for d_meth in detect_methods:
                if d_meth not in self.DETEC_METHOD:
                    raise ValueError(f"The detection method {d_meth} does not exist in {self.DETEC_METHOD}")
        if replace_method is not None:
            for r_meth in replace_method:
                if r_meth not in self.REPL_METHOD:
                    raise ValueError(f"The replace method {r_meth} does not exist in {self.REPL_METHOD}")
        if detect_methods is None:
            self._detect_method = ["standard_deviation"]
        else:
            self._detect_method = detect_methods
        if replace_method is None:
            self._replace_method = ["median"]
        else:
            self._replace_method = replace_method

    @property
    def serie(self) -> pd.Series:
        """
        Series is the timesseries for which we want to find the outliers
        """
        return self._serie

    @serie.setter
    def serie(self, values: pd.Series):
        if not isinstance(values, pd.Series):
            raise TypeError(f"The type of the series must be pd.Series and not {type(values)}")
        self._serie = values

    @property
    def detect_method(self) -> List[str]:
        """
        "detect_method" is the list containing the detect methods with which we want to find the outliers
        """
        return self._detect_method

    @property
    def replace_method(self) -> List[str]:
        """
        "replace_method" is the list containing the replace methods with which we want to replace the outliers
        """
        return self._replace_method

    def get_zscore(self, value) -> int:
        """
        Computes the zscore of "value". "value" corresponding to a value of self._serie.

        Parameters
        ----------
        value : int

        Returns
        -------
        int
            zscore of value
        """
        mean = np.mean(self.serie)
        std = np.std(self.serie)
        z_score = (value - mean) / std
        return np.abs(z_score)

    def _verif_norm(self, alpha_level=0.05):
        serie = self.serie.dropna()
        print(serie, "hjunnnnifcccc")
        df_res = pd.DataFrame(serie)
        ks = stats.ks_1samp(serie, stats.norm.cdf)
        print(ks)
        df_res.loc["p-value", serie.name] = ks.pvalue
        if df_res.loc["p-value", serie.name] < alpha_level:
            return True
        else:
            return False

    def _zscore(self) -> pd.Series:
        """
        This method detects outliers from the z_score

        Returns
        -------
        pd.Series
            outliers is a series containing date and value of outliers
        """
        serie = self.serie.dropna()
        res = pd.DataFrame(serie)
        res["z-score"] = serie.apply(lambda x: self.get_zscore(x))
        res = res[res["z-score"] > 3]
        outliers = res[serie.name]
        return outliers

    def _iqr(self) -> pd.Series:
        """
        This method entails using the 1st quartile, 3rd quartile, and IQR to define the lower bound and upper bound
        for the data points.

        Returns
        -------
        pd.Series
            outliers is a series containing date and value of outliers
        """
        serie = self.serie.dropna()
        quantile_1 = np.quantile(serie, 0.25)
        quantile_3 = np.quantile(serie, 0.75)
        iqr = quantile_3 - quantile_1
        lower_bound = quantile_1 - 1.5 * iqr
        upper_bound = quantile_3 + 1.5 * iqr

        outliers = serie[(serie < lower_bound) | (serie > upper_bound)]
        return outliers

    def _standard_deviation(self) -> pd.Series:
        """
        This method find the outliers with the standard deviation

        Returns
        -------
        pd.Series
            outliers is a series containing date and value of outliers
        Raises
        -------
            ValueError
        """
        if self._verif_norm() is False:
            # TODO maybe change the type of error
            raise ValueError("The series does not follow a normal law, so we cannot use this method")
        serie = self.serie.dropna()
        mean, std = np.mean(serie), np.std(serie)
        cut_off = std * 3
        lower, upper = mean - cut_off, mean + cut_off
        outliers = serie[(serie < lower) | (serie > upper)]

        return outliers

    # TODO pytest @eguir
    def _detect_outliers(self) -> List[list]:
        """
        This method detects outliers based on self._detetct_method

        Returns
        -------
        list
            list containing the outliers according to the method
        """
        output = []
        for m in self.detect_method:
            output.append(getattr(self, f"_{m}"))
        return output

    def _minimum(self, minimum: int) -> List:
        """
        Detects outliers with respect to a given minimum

        Parameters
        ----------
        minimum : int

        Returns
        -------
        list
             list containing the outliers
        """
        pass

    def _maximum(self, maximum: int) -> List:
        """
        Detects outliers with respect to a given maximum

        Parameters
        ----------
        maximum : int

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    # TODO pyetst @eguir
    def replace_by_median(self, outliers) -> pd.Series:
        """

        Parameters
        ----------
        outliers :

        Returns
        -------

        """
        self.serie.dropna(inplace=True)
        median = np.median(self.serie)
        self.serie.loc[outliers.index] = median
        return self.serie

    # TODO pyetst @eguir
    def replace_by_post_value(self, ouliers):
        self.serie.dropna(inplace=True)
        df = self.serie.reset_index()
        for i in ouliers.index:
            index = df.loc[df["index"] == i].index.item() - 1
            value = df.loc[index][1]
            df.loc[df["index"] == i] = value
            self.serie.loc[i] = value
        return self.serie

    def _gaussian_mixture_model(self):
        """
        A Gaussian mixture model is a probabilistic model that assumes that all data points are generated
        from a mixture of a finite number of Gaussian distributions with unknown parameters.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _extreme_value_analysis(self):
        """
        Estimation of the probability of the rarest events compared to those previously compared.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _local_outlier_factor(self):
        """
        In anomaly detection, the local outlier factor is an algorithm that finds anomalous data points
        by measuring the local deviation of a given data point from its neighbours.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _connectivity_based_outlier_detection(self):
        """
        The connectivity-based outlier detection is a technique for detecting outliers. It is an improved version of the
        local outlier factor (LOF) technique. The idea of the connectivity-based outlier algorithm is to assign a degree
        of outlier to each data point. This degree of outlier is called the connectivity-based outlier factor (COF) of
        the data point. A high COF of a data point represents a high probability of being an outlier.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _angular_based_outlier_detection(self):
        """
        The approach called ABOD (Angle-based Outlier Detection) evaluates the degree of outlier on the variance of
        angles (VOA) between a point and all other pairs of points in the data set.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _dbscan_clustering(self):
        """
        Clustering is a way to group a set of data points in a way that similar data points are grouped together.
        Therefore, clustering algorithms look for similarities or dissimilarities among data points. Clustering is an
        unsupervised learning method so there is no label associated with data points. The algorithm tries to find the
        underlying structure of the data.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _kmeans_clustering(self):
        """
        Groups data points into k clusters based on their feature values. The scores of each data point within a cluster
        are calculated as the distance to its centroid. Data points that are far from the centroid of their clusters are
        labelled as anomalies.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _knearest_neighbor(self):
        """
        For each data point, the whole set of data points is examined to extract the k items that have the most similar
        feature values: these are the k nearest neighbors (NN). Then, the data point is classified as anomalous if the
        majority of NN was previously classified as anomalous.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _mahalanobis_distance(self):
        """
        Calculates the distance to the barycentre taking into account the shape of all data points. In an area of high
        density, a point that deviates from the others (its immediate neighbours) should raise more questions than when
        it is located in a less dense area.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _isolation_forest(self):
        """
        The isolation forest attempts to separate each point from the data. In the case of 2D,
        it randomly creates a line
        and tries to isolate a point. In this case, an abnormal point can be separated in a few steps, while normal
        points that are closer together may take many more steps to separate.

        Returns
        -------
        list
            List containing the outliers
        """
        pass

    def _robust_random_cut_forest(self):
        pass

    def _support_vector_machine(self):
        """
        One-class Support Vector Machine algorithm aims at learning a decision boundary to group the data points. Each
        data point is classified considering the normalized distance of the data point from the determined decision
        boundary.

        Returns
        -------
        list
            List containing the outliers
        """
        pass


#
# df = pd.read_csv("/home/eguin/PycharmProjects/adclean/notebook/gm_matrix.csv", index_col=0)
# # print(df.iloc[::, :1])
#
# serie = df
# serie2 = df["SPREAD_GOV_JPN_1Y-GOV_GER_1Y_Z250D"]
# # print(type(serie2.index[0]))
# serie2.hist()
# plt.show()
# #
# c1 = Cleaner(serie2)
# # print("aaaa", len(c1._z_score()), "BBBBB")
# # print("TEST", len(c1._standard_deviation()), "TEST14")
#
# res = c1._standard_deviation()
# print("SANS REPLACE", res)
# res2 = c1.replace_by_post_value(res)
# print("AVEC REPLACE", res2)
# #
# c2 = Cleaner(res2)
# print(c2._zscore())

"---------------"
#
# c3 = Cleaner(serie=pd.Series(
#     [10, 12, 14, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 21, 24, 35, 39, 21, 45, 159, 180],
#     index=pd.date_range(start="2022-01-01", end="2022-01-25"), name="value"))
#
# print(c3.verif_norm())
length = 3000
data = np.random.gamma(1, 1, 5)
ser = pd.Series(data, index=pd.date_range(start="2022-02-12", end="2022-02-16"), name="value")
ser.hist()
plt.show()
c4 = Cleaner(ser)
print(c4._verif_norm())
# print(pd.Series(data, index=pd.date_range(start="2013-12-01", end="2022-02-16"), name="value"))
# print(type(data))

# print(res)
# print(c1._z_score(), "------", len(c1._z_score()))
# res = c1._z_score().index
#
# serie2.drop(res, inplace=True)
# serie2.plot()
# plt.show()

# df = pd.read_csv("/home/eguin/PycharmProjects/adclean/notebook/gm_matrix.csv", index_col=0)
# print(df.iloc[::, :1])

# serie = df
# serie2 = df["BOVES_DY 12M_DSTD20_100"].dropna()
# serie2 = serie2.replace(0, np.nan)
# serie2.fillna(method="ffill", inplace=True)
# # serie2.replace(0, np.nan, inplace=True)
# serie2.plot()
# plt.show()
# #
# c1 = Cleaner(serie2)
#
# print(c1._z_score(), "------", len(c1._z_score()))
