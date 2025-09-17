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

    def __init__(self, embs:List[np.array]=None, cluster_labels = None, series:np.array = None, change_metric=None, timeseries_type:str = None, k=1, time_labels : Union[np.array, List] = None, clustering_algorithm = None, distance_metric='cosine'):
        # Init from embeddings
        if embs is not None:
            self.compute(embs=embs, cluster_labels=None, change_metric=change_metric, timeseries_type=timeseries_type, k=k, time_labels=time_labels, clustering_algorithm=clustering_algorithm, distance_metric=distance_metric)
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
            Args:
                embs ([np.array]): a list of embeddings, each element of the list contains embeddings from one time period.
                cluster_labels ([np.array]): a list of arrays, each array contains the cluster labels for the embeddings in one time period (used only if using PJSD as change metric).
                change_metric (str|object): the metric to use when comparing embeddings from different time periods (should be one of the classes in languagechange.models.change.metrics).
                timeseries_type (str): the kind of timeseries to construct. One of ['compare_to_first', 'compare_to_last', 'consecutive', 'moving_average'].
                time_labels (np.array|list): labels for the x axis of the timeseries.
                clustering_algorithm: the clustering algorithm if using PJSD as the change metric. E.g. one of the algorithms in scikit-learn, or languagechange.
                distance_metric (str): the distance metric to use when computing change scores.
            Returns:
                series (np.array): the final timeseries.
                ts (np.array): the time values/labels for each value in the final timeseries.
        """
        if type(change_metric) == str:
            try:
                change_metric = {'apd': APD(), 'prt': PRT(), 'pjsd': PJSD()}[change_metric.lower()]
            except:
                logging.error("Error: if 'change_metric' is a string it must be one of 'apd','prt' and 'pjsd'.")
                raise Exception
            
        if not isinstance(change_metric, GradedChange):
            logging.error("Error: if 'change_metric' is an object it must be an instance of GradedChange.")
            raise Exception
        
        if isinstance(change_metric, PJSD):
            if embs is not None:
                compute_scores = lambda e1, e2 : change_metric.compute_scores(e1, e2, clustering_algorithm, distance_metric)
            elif cluster_labels is not None:
                compute_scores = lambda l1, l2 : change_metric.compute_scores_from_labels(l1, l2)
            else:
                logging.error("Error: if using PJSD as change metric, either 'embs' or 'cluster_labels' must be provided.")
                raise Exception
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
        elif timeseries_type == "moving_average":
            series = ma(np.array([compute_scores(data[i],data[i+1]) for i in range(len(data)-1)]), k)
            t_idx = np.array(range(k+1,len(data)-k))

        if time_labels is not None:
            ts = np.array(time_labels)[t_idx]
        else:
            ts = t_idx

        self.series = series
        self.ts = ts
        return series, ts