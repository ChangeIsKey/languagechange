from typing import List, Union
import numpy as np
from languagechange.models.change.metrics import GradedChange, APD, PRT, JSD
import logging


def moving_average(ts, k):
    """
        Computes the moving average of a timeseries.
        Args:
            ts (np.array) : a timeseries.
            k (int) : the window (k timesteps to the left and k to the right)
        Returns:
            the moving average of the timeseries (not including endpoints)
    """
    return np.convolve(ts, np.ones(2*k+1))[2*k:-2*k] / (2*k+1)


class TimeSeries:
    """
        Handles time series of embeddings or cluster labels and computes change scores between them. See 
        languagechange.models.change.metrics.JSD for change score computation from cluster labels. Change scores can
        be computed by comparing each value to the first or the last in the time series or by comparing adjacent values
        to each other with the possibility to apply a moving average.

        A TimeSeries object can also be initialized with an already defined time series.

        Labels for the time axis can also be added as a list or array.

        Parameters:
            embeddings_or_cluster_labels ([Union[None, np.array, List]], default=None): a list of either embeddings (as 
                a 2d array) or cluster labels (as a 1d array or list), one element for each time period.
            series (np.array, default=None): an already defined time series. If both embs and cluster_labels are None,
                but this is not, it will be used.
            change_metric (str|object, default=None): the metric to use when comparing embeddings from different time 
                periods (should be one of the classes in languagechange.models.change.metrics).
            timeseries_type (str, default=None): the kind of timeseries to construct. One of ['compare_to_first', 
                'compare_to_last', 'consecutive', 'moving_average'].
            k (int, default=1): window size, if moving average is applied.
            time_labels (np.array|list): labels for the x axis of the timeseries.
            clustering_algorithm: the clustering algorithm if using JSD as the change metric. E.g. one of the 
                algorithms in scikit-learn, or languagechange.
            distance_metric (str, default="cosine"): the distance metric to use when computing change scores.
    """

    def __init__(self, embeddings_or_cluster_labels=None, series: np.array = None, change_metric=None,
                 timeseries_type: str = None, k=1, time_labels: Union[np.array, List] = None, 
                 clustering_algorithm=None, distance_metric='cosine'):
        # Init from embeddings or cluster labels
        if embeddings_or_cluster_labels:
            self.compute(embeddings_or_cluster_labels, change_metric=change_metric, timeseries_type=timeseries_type,
                         k=k, time_labels=time_labels, clustering_algorithm=clustering_algorithm, 
                         distance_metric=distance_metric)
        # Init from an already constructed timeseries
        elif series is not None:
            self.series = series
            if time_labels is not None:
                self.ts = time_labels[self.series]
        else:
            self.series = np.array([])

    def compute(self,
                embeddings_or_cluster_labels,
                change_metric: Union[str, object],
                timeseries_type: str,
                k=1,
                time_labels: Union[np.array, List] = None,
                clustering_algorithm=None,
                distance_metric: str = 'cosine',
                return_labels=False):
        """
            Computes the change scores for each point in the time series, using either embeddings or cluster labels
            depending on the change metric (JSD can start from embeddings or cluster labels while APD and PRT involve
            embeddings only).

            Args:
                embeddings_or_cluster_labels ([Union[np.array, List]], default=None): a list of either embeddings (as 
                    a 2d array) or cluster labels (as a 1d array or list), one element for each time period.
                cluster_labels ([np.ndarray], default=None): a list of arrays, each array contains the cluster labels 
                    for the embeddings in one time period (used only if using JSD as change metric).
                change_metric (Union[str, GradedChange], default=None): the metric to use when comparing embeddings from different 
                    time periods (should be one of the classes in languagechange.models.change.metrics).
                timeseries_type (str, default=None): the kind of timeseries to construct. One of ['compare_to_first', 
                    'compare_to_last', 'consecutive', 'moving_average'].
                k (int, default=1): window size, if moving average is applied.
                time_labels (Union[np.ndarray, List]): labels for the x axis of the timeseries.
                clustering_algorithm: the clustering algorithm if using JSD as the change metric. E.g. one of the 
                    algorithms in scikit-learn, or languagechange.
                distance_metric (str, default="cosine"): the distance metric to use when computing change scores.
                return_labels (bool, default=False): whether to return cluster labels (only applicable for JSD).
            Returns:
                series (np.ndarray): the final timeseries.
                ts (np.ndarray): the time values/labels for each value in the final timeseries.
                labels (List[tuple[np.ndarray]], optional): cluster labels, sorted by time.
        """
        if timeseries_type not in {"compare_to_first", "compare_to_last", "consecutive", "moving_average"}:
            logging.error(
                "'time_series_type' must be one of 'compare_to_first', 'compare_to_last', 'consecutive', and "
                "'moving_average'")
            raise ValueError

        if isinstance(change_metric, str):
            try:
                change_metric = {'apd': APD(), 'prt': PRT(), 'jsd': JSD()}[change_metric.lower()]
            except KeyError as e:
                logging.error("Error: if 'change_metric' is a string it must be one of 'APD','PRT' and 'JSD'.")
                raise ValueError from e

        if not isinstance(change_metric, GradedChange):
            logging.error("Error: if 'change_metric' is an object it must be an instance of GradedChange.")
            raise TypeError

        if isinstance(change_metric, JSD):
            # Compute from embeddings
            if all(isinstance(e, np.ndarray) and e.ndim == 2 for e in embeddings_or_cluster_labels):
                if not clustering_algorithm:
                    logging.error(
                        "For computing change scores from embeddings with JSD, `clustering_algorithm` needs to be "
                        "provided.")
                    raise ValueError
                def compute_scores(e1, e2):
                    return change_metric.compute_scores(e1,
                                                        e2,
                                                        clustering_algorithm,
                                                        distance_metric,
                                                        return_labels=return_labels)
            # Compute from cluster labels
            elif all((isinstance(cl, np.ndarray) and cl.ndim == 1) or
                     isinstance(cl, list) for cl in embeddings_or_cluster_labels):
                compute_scores = change_metric.compute_scores_from_labels
            else:
                logging.error(
                    "Error: if using JSD as change metric, 'embeddings_or_cluster_labels' must be a list of \n"
                    "* 2d np.ndarray containing embeddings, or\n"
                    "* a 1d np.ndarray or list containing cluster labels.")
                raise ValueError
        else:
            if not all(isinstance(e, np.ndarray) and e.ndim == 2 for e in embeddings_or_cluster_labels):
                logging.error(
                    f"Error: if using {type(change_metric).__name__} as change metric, 'embeddings_or_cluster_labels' "
                     "must be a list of 2d np.ndarray containing embeddings.")
            def compute_scores(e1, e2):
                return change_metric.compute_scores(e1, e2, distance_metric)

        # Compare every time period with the first one
        if timeseries_type == "compare_to_first":
            scores = [compute_scores(embeddings_or_cluster_labels[0], d) for d in embeddings_or_cluster_labels[1:]]
            t_idx = np.array(range(1, len(embeddings_or_cluster_labels)))

        # Compare every time period with the last one
        elif timeseries_type == "compare_to_last":
            scores = [compute_scores(d, embeddings_or_cluster_labels[-1]) for d in embeddings_or_cluster_labels[:-1]]
            t_idx = np.array(range(len(embeddings_or_cluster_labels)-1))

        # Compare consecutive time periods
        elif timeseries_type == "consecutive":
            scores = [compute_scores(embeddings_or_cluster_labels[i],
                                     embeddings_or_cluster_labels[i + 1])
                      for i in range(len(embeddings_or_cluster_labels) - 1)]
            t_idx = np.array(range(1, len(embeddings_or_cluster_labels)))

        # Moving average
        else:
            scores = [compute_scores(embeddings_or_cluster_labels[i],
                                     embeddings_or_cluster_labels[i + 1])
                      for i in range(len(embeddings_or_cluster_labels) - 1)]
            t_idx = np.array(range(k+1, len(embeddings_or_cluster_labels)-k))

        if return_labels:
            series, labels = zip(*scores)
            series = np.array(series)
        else:
            series = np.array(scores)

        if timeseries_type == "moving_average":
            series = moving_average(series, k)

        if time_labels is not None:
            ts = np.array(time_labels)[t_idx]
        else:
            ts = t_idx

        self.series = series
        self.ts = ts
        if return_labels:
            return series, ts, list(labels)
        return series, ts
