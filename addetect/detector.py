from typing import List, Optional
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

import logging

logger = logging.getLogger(__name__)


class Detector:
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

    # TODO : Verif the type of the index @eguir HINT (pcotte): use 'adtypingdecorator'
    def __init__(
            self,
            serie: Optional[pd.Series] = None,
            detect_methods: Optional[List[str]] = None,
    ):
        """
        The constructor of the Detector class.

        Parameters
        ----------
        serie : pd.Series
            If None, will have to be specified later by setting the instance's 'series' attribute.
        detect_methods : List[str]
            If None, will default to 'standard_deviation'.

        Raises
        ---------
        TypeError
            If 'series' is not a pd.Series object.
        ValueError
            If any of the detection methods is unknown by the class.
        """
        if not isinstance(serie, pd.Series):
            raise TypeError(f"The type of the series must be pd.Series and not {type(serie)}")
        self._serie = serie
        if detect_methods is not None:
            for d_meth in detect_methods:
                if d_meth not in self.DETEC_METHOD:
                    raise ValueError(f"The detection method {d_meth} does not exist in {self.DETEC_METHOD}")
        if detect_methods is None:
            self._detect_method = ["standard_deviation"]
        else:
            self._detect_method = detect_methods

    @property
    def serie(self) -> pd.Series:
        """
        Series is the timesseries for which we want to find the outliers.
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
        "detect_method" is the list containing the detect methods with which we want to find the outliers.
        """
        return self._detect_method

    def get_zscore(self, value) -> int:
        """
        Computes the zscore of "value". ("value" corresponding to a value of self._serie).

        Parameters
        ----------
        value : int
            The value whose zscore is desired.
        Returns
        -------
        int
            zscore of value.
        """
        mean = np.mean(self.serie)
        std = np.std(self.serie)
        z_score = (value - mean) / std
        return np.abs(z_score)

    def _verif_norm(self, alpha_level=0.05):
        """
        This method checks if the series follows a normal distribution.

        Parameters
        ----------
        alpha_level : float
            the probability of making the wrong decision.

        Returns
        -------
        bool
            True if p-value < alpha_lever
            False else
        """
        serie = self.serie.dropna()
        df_res = pd.DataFrame(serie)
        ks = stats.ks_1samp(serie, stats.norm.cdf)
        df_res.loc["p-value", serie.name] = ks.pvalue
        if df_res.loc["p-value", serie.name] < alpha_level:
            return True
        else:
            return False

    def _zscore(self) -> pd.Series:
        """
        This method detects outliers from the z_score.

        Returns
        -------
        pd.Series
            outliers is a series containing date and value of outliers.
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
            # TODO maybe change the type of error @eguir
            raise ValueError("The series does not follow a normal law, so we cannot use this method")
        serie = self.serie.dropna()
        mean, std = np.mean(serie), np.std(serie)
        cut_off = std * 3
        lower, upper = mean - cut_off, mean + cut_off
        outliers = serie[(serie < lower) | (serie > upper)]

        return outliers

    def _isolation_forest(self, outlier_fraction=.01) -> pd.Series:
        """
        The isolation forest attempts to separate each point from the data. In the case of 2D,
        it randomly creates a line
        and tries to isolate a point. In this case, an abnormal point can be separated in a few steps, while normal
        points that are closer together may take many more steps to separate.

        Returns
        -------
        pd.Series
            Series containing the outliers
        """
        scaler = StandardScaler()
        serie = self.serie.dropna()
        df_res = pd.DataFrame(serie)
        np_scaled = scaler.fit_transform(serie.values.reshape(-1, 1))
        data = pd.DataFrame(np_scaled)
        model = IsolationForest(contamination=outlier_fraction)
        model.fit(data)
        df_res['anomaly'] = model.predict(data)
        a = df_res.loc[df_res['anomaly'] == -1, [serie.name]]  # anomaly
        print(a)
        return a

    # TODO pytest @eguir
    def _detect_outliers(self) -> List[list]:
        """
        This method detects outliers based on self._detetct_method.

        Returns
        -------
        list
            list containing the outliers according to the different method.
        """
        output = []
        for m in self.detect_method:
            output.append(getattr(self, f"_{m}"))
        return output

    # TODO pytest
    def _first_date(self):
        """
        This method finds the first non-null date.

        Returns
        -------
        str
            The first non-null date
        """
        return self.serie.first_valid_index()

    # TODO pytest
    def _last_date(self):
        """
        This method finds the last non-null date.

        Returns
        -------
        str
            The last indenon null date.

        """
        return self.serie.last_valid_index()

    # TODO pytest
    def _count_date(self):
        """
        Count the number of dates from the first to the last non-null index

        Returns
        -------
        int
            Number of date
        """

        return len(self.serie.loc[self._first_date():self._last_date()])

    # TODO pytest
    def _serie_between_first_and_last_index(self):
        """
        Get the serie between the first and last date non-null

        Returns
        -------
        pd.Series

        """
        return self.serie.loc[self._first_date():self._last_date()]

    # TODO pytest
    def _count_nan_between_index(self):
        """
        Count the number of nan between the first and last index

        Returns
        -------
        int
            number of nan
        """
        return self.serie.loc[self._first_date():self._last_date()].isna().sum()

    # TODO pytest
    def _not_jump_date(self, freq="B"):
        """
        Check that no dates are missing according to the frequency

        Returns
        -------
        bool
            True If there are missing dates
            False else

        """
        return len(pd.date_range(start=self._first_date(), end=self._last_date(), freq=freq)) == len(
            self._serie_between_first_and_last_index())

    # TODO pytest
    def verif_not_duplicate_index(self):
        """
        Check that there are no duplicate indexes.

        Returns
        -------
        True If there is no
        False else

        """
        return self.serie.index.duplicated().sum() == 0

    # TODO pytest
    def _get_minimum(self) -> float:
        """
        Get the minimum of the serie.

        Returns
        -------
        float
            The minimum of the serie
        """
        return min(self.serie)

    # TODO pytest
    def _get_maximum(self) -> List:
        """
        Get the maximum of the serie.

        Returns
        -------
        float
            The maximum of the serie
        """
        return max(self.serie)

    # TODO pytest
    def _get_standard_deviation(self) -> float:
        """
        Get the standard deviation of the serie.

        Returns
        -------
        float
            The standard deviation of the serie
        """
        return float(np.std(self.serie))

    # TODO pytest
    def _get_mean(self) -> float:
        """
        Get the standard deviation of the serie.

        Returns
        -------
        float
            The standard deviation of the serie
        """
        return float(np.mean(self.serie))

    # TODO pytest
    def _get_first_quantile(self):
        """
        Get the first quantile of the serie.

        Returns
        -------
        float
            The first quantile

        """
        return np.quantile(self.serie.dropna(), 0.25)

    # TODO pytest
    def _get_last_quantile(self):
        """
        Get the first quantile of the serie.

        Returns
        -------
        float
            The first quantile

        """
        return np.quantile(self.serie.dropna(), 0.75)

    # TODO pytest
    def _get_median(self):
        """
        Get the median of the serie.

        Returns
        -------
        float
            median of serie
        """
        return np.median(self.serie.dropna())

    # TODO pytest
    def _verif_type_of_serie(self, type):
        """
        Check that the data type of our series matches the one in parameter

        Returns
        -------
        float
            median of serie
        """

        if type is None:
            raise ValueError("The type cannot be null")
        try:
            return isinstance(self.serie.dtypes, type)
        except TypeError:
            raise TypeError(f"The type {type} does not exist")

    # TODO pytest
    def _variation_between_date(self) -> pd.Series:
        """

        Returns
        -------
        pd.Series
            The serie with the pct change for each value
        """
        return self.serie.dropna().pct_change()

    # TODO pytest
    def _max_variation(self) -> float:
        """

        Returns
        -------

        """

        return max(self._variation_between_date().dropna())

    # TODO pytest
    def _min_variation(self) -> float:
        """

        Returns
        -------

        """

        return min(self._variation_between_date().dropna())

    def _plot_with_outliers(self, outliers: pd.Series, show: bool = False):
        """
        The graph of the data as a plot, and the outliers as a scatter.
        Parameters
        ----------
        outliers : pd.Serie
            The serie of outliers
        """
        serie = self.serie.dropna()
        plt.figure(figsize=(20, 10))
        plt.xlabel("Date")
        plt.ylabel(serie.name)
        plt.plot(pd.DatetimeIndex(serie.index), serie.values)
        plt.scatter(pd.DatetimeIndex(outliers.index), outliers.values, c="red")
        plt.savefig("plot/plot")
        if show:
            plt.show()

