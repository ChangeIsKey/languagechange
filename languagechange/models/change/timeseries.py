from typing import List, Union
import numpy as np
from languagechange.models.change.metrics import GradedChange, APD, PRT, PJSD
import logging


def ma(ts, k):
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
        languagechange.models.change.metrics.PJSD for change score computation from cluster labels. Change scores can
        be computed by comparing each value to the first or the last in the time series or by comparing adjacent values
        to each other with the possibility to apply a moving average.

        A TimeSeries object can also be initialized with an already defined time series.

        Labels for the time axis can also be added as a list or array.

        Parameters:
            embs ([np.array], default=None): a list of embeddings, each element of the list contains embeddings from 
                one time period.
            cluster_labels ([np.array], default=None): a list of arrays, each array contains the cluster labels for the 
                embeddings in one time period (used only if using PJSD as change metric).
            series (np.array, default=None): an already defined time series. If both embs and cluster_labels are None,
                but this is not, it will be used.
            change_metric (str|object, default=None): the metric to use when comparing embeddings from different time 
                periods (should be one of the classes in languagechange.models.change.metrics).
            timeseries_type (str, default=None): the kind of timeseries to construct. One of ['compare_to_first', 
                'compare_to_last', 'consecutive', 'moving_average'].
            k (int, default=1): window size, if moving average is applied.
            time_labels (np.array|list): labels for the x axis of the timeseries.
            clustering_algorithm: the clustering algorithm if using PJSD as the change metric. E.g. one of the 
                algorithms in scikit-learn, or languagechange.
            distance_metric (str, default="cosine"): the distance metric to use when computing change scores.
    """

    def __init__(self, embs:List[np.array]=None, cluster_labels = None, series:np.array = None, change_metric=None, timeseries_type:str = None, k=1, time_labels : Union[np.array, List] = None, clustering_algorithm = None, distance_metric='cosine'):
        # Init from embeddings
        if embs is not None:
            self.compute(embs=embs, cluster_labels=None, change_metric=change_metric, timeseries_type=timeseries_type, k=k, time_labels=time_labels, clustering_algorithm=clustering_algorithm, distance_metric=distance_metric)
        # Init from cluster labels (for PJSD)
        elif cluster_labels is not None:
            self.compute(embs=None, cluster_labels=cluster_labels, change_metric=change_metric, timeseries_type=timeseries_type, k=k, time_labels=time_labels, clustering_algorithm=clustering_algorithm, distance_metric=distance_metric)
        # Init from an already constructed timeseries
        elif series is not None:
            self.series = series
            if time_labels is not None:
                self.ts = time_labels[self.series]
        else:
            self.series = np.array([])

    def compute(self, embs : List[np.array] = None, cluster_labels = None, change_metric : Union[str, object] = None, timeseries_type : str = None, k=1, time_labels : Union[np.array, List] = None, clustering_algorithm = None, distance_metric : str = 'cosine'):
        """
            Computes the change scores for each point in the time series, using either embeddings or cluster labels
            depending on the change metric (PJSD can start from embeddings or cluster labels while APD and PRT involve
            embeddings only).

            Args:
                embs ([np.array], default=None): a list of embeddings, each element of the list contains embeddings 
                    from one time period.
                cluster_labels ([np.array], default=None): a list of arrays, each array contains the cluster labels 
                    for the embeddings in one time period (used only if using PJSD as change metric).
                change_metric (str|object, default=None): the metric to use when comparing embeddings from different 
                    time periods (should be one of the classes in languagechange.models.change.metrics).
                timeseries_type (str, default=None): the kind of timeseries to construct. One of ['compare_to_first', 
                    'compare_to_last', 'consecutive', 'moving_average'].
                k (int, default=1): window size, if moving average is applied.
                time_labels (np.array|list): labels for the x axis of the timeseries.
                clustering_algorithm: the clustering algorithm if using PJSD as the change metric. E.g. one of the 
                    algorithms in scikit-learn, or languagechange.
                distance_metric (str, default="cosine"): the distance metric to use when computing change scores.
            Returns:
                series (np.array): the final timeseries.
                ts (np.array): the time values/labels for each value in the final timeseries.
        """
        if timeseries_type not in {"compare_to_first", "compare_to_last", "consecutive", "moving_average"}:
            logging.error("'time_series' must be one of 'compare_to_first', 'compare_to_last', 'consecutive', and 'moving_average'")
            raise ValueError
        
        if isinstance(change_metric, str):
            try:
                change_metric = {'apd': APD(), 'prt': PRT(), 'pjsd': PJSD()}[change_metric.lower()]
            except:
                logging.error("Error: if 'change_metric' is a string it must be one of 'APD','PRT' and 'PJSD'.")
                raise ValueError
            
        if not isinstance(change_metric, GradedChange):
            logging.error("Error: if 'change_metric' is an object it must be an instance of GradedChange.")
            raise TypeError
        
        if isinstance(change_metric, PJSD):
            if embs is not None:
                compute_scores = lambda e1, e2 : change_metric.compute_scores(e1, e2, clustering_algorithm, distance_metric)
            elif cluster_labels is not None:
                compute_scores = change_metric.compute_scores_from_labels
            else:
                logging.error("Error: if using PJSD as change metric, either 'embs' or 'cluster_labels' must be provided.")
                raise ValueError
        else:
            compute_scores = lambda e1, e2 : change_metric.compute_scores(e1, e2, distance_metric)

        data = embs if embs is not None else cluster_labels
        
        # Compare every time period with the first one
        if timeseries_type == "compare_to_first":
            series = np.array([compute_scores(data[0],d) for d in data[1:]])
            t_idx = np.array(range(1,len(data)))

        # Compare every time period with the last one
        elif timeseries_type == "compare_to_last":
            series = np.array([compute_scores(d,data[-1]) for d in data[:-1]])
            t_idx = np.array(range(len(data)-1))

        # Compare consecutive time periods
        elif timeseries_type == "consecutive":
            series = np.array([compute_scores(data[i],data[i+1]) for i in range(len(data)-1)])
            t_idx = np.array(range(1, len(data)))

        # Moving average
        else:
            series = ma(np.array([compute_scores(data[i],data[i+1]) for i in range(len(data)-1)]), k)
            t_idx = np.array(range(k+1,len(data)-k))

        if time_labels is not None:
            ts = np.array(time_labels)[t_idx]
        else:
            ts = t_idx

        self.series = series
        self.ts = ts
        return series, ts