from typing import List
import pandas as pd


class Cleaner:
    detec_method = ["standard_deviation", "gaussian_mixture_model", "extreme_value_analysis", "local_outlier_factor",
                    "connectivity_based_outlier_detection", "angular_based_outlier_detection", "dbscan_clustering",
                    "kmeans_clustering", "knearest_neighbor", "mahalanobis_distance", "isolation_forest",
                    "support_vector_machine", "minimum", "maximum"]

    # TODO : Add the different replace methods
    repl_method = ["median"]

    # TODO : Verif the type of the index @eguir
    def __init__(self, serie: pd.Series = None, detect_method: List[str] = None,
                 replace_method: List[str] = None):
        """
        The constructor of the Cleaner class, by default the series can be None but will have to be entered later
        using the setter if detect_method is not entered then by default it will use the standard_deviation method,
        the replacement method can be empty but by default it will take the replacement method mean

        Parameters
        ----------
        serie : pd.Series
        detect_method : List[str]
        replace_method : List[str]

        Raises
        ---------
        ValueError, TypeError
        """
        if type(serie) != pd.Series:
            raise TypeError(f"The type of the series must be pd.Series and not {type(serie)}")
        self._serie = serie
        if detect_method is not None:
            for d_meth in detect_method:
                if d_meth not in self.detec_method:
                    raise ValueError(f"The detection method {d_meth} does not exist in {self.detec_method}")
        if replace_method is not None:
            for r_meth in replace_method:
                if r_meth not in self.repl_method:
                    raise ValueError(f"The replace method {r_meth} does not exist in {self.repl_method}")
        if detect_method is None:
            self._detect_method = ["standard_deviation"]
        else:
            self._detect_method = detect_method
        if replace_method is None:
            self._replace_method = ["median"]
        else:
            self._replace_method = replace_method

    @property
    def serie(self) -> pd.Series:
        """
        Returns
        -------
        pd.Series
        """
        return self._serie

    @serie.setter
    def serie(self, values: pd.Series):
        """

        Parameters
        ----------
        values : pd.Series

        Raises
        -------
        TypeError
        """
        if type(values) != pd.Series:
            raise TypeError(f"The type of the series must be pd.Series and not {type(values)}")
        self._serie = values

    @property
    def detect_method(self) -> List[str]:
        """
        Getter from detect_method attribut

        Returns
        -------
        List[str]
        """
        return self._detect_method

    @property
    def replace_method(self) -> List[str]:
        """
        Getter from replace_method attribut

        Returns
        -------
        List[str]
        """
        return self._replace_method

    def _detect_outliers(self) -> List[list]:
        """
        This method detects outliers based on the method entered the argument self._detetct_method

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

    def _standard_deviation(self):
        """
        Points within 3 standard deviations of the mean constitute only about 1% of the distribution.
        These points are atypical of the majority of the other points and are likely to be outliers.

        Returns
        -------
        List[outliers]
            List containing the outliers
        """
        pass

    def _gaussian_mixture_model(self):
        """
        A Gaussian mixture model is a probabilistic model that assumes that all data points are generated
        from a mixture of a finite number of Gaussian distributions with unknown parameters.

        Returns
        -------
        List[outliers]
            List containing the outliers
        """
        pass

    def _extreme_value_analysis(self):
        """
        Estimation of the probability of the rarest events compared to those previously compared.

        Returns
        -------
        List[outliers]
            List containing the outliers
        """
        pass

    def _local_outlier_factor(self):
        """
        In anomaly detection, the local outlier factor is an algorithm that finds anomalous data points
        by measuring the local deviation of a given data point from its neighbours.

        Returns
        -------
        List[outliers]
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
        List[outliers]
            List containing the outliers
        """
        pass

    def _angular_based_outlier_detection(self):
        """
        The approach called ABOD (Angle-based Outlier Detection) evaluates the degree of outlier on the variance of
        angles (VOA) between a point and all other pairs of points in the data set.

        Returns
        -------
        List[outliers]
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
        List[outliers]
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
        List[outliers]
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
        List[outliers]
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
        List[outliers]
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
        List[outliers]
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
        List[outliers]
            List containing the outliers
        """
        pass
