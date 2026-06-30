from languagechange.models.meaning.clustering import Clustering, APosterioriaffinityPropagation
from languagechange.models.change.timeseries import TimeSeries
import numpy as np
from typing import List, Union


class WiDiD:
    """
        A class that implements the What-is-Done-is-Done (WiDiD) algorithm (https://github.com/FrancescoPeriti/WiDiD).
    """

    def __init__(self, algorithm=APosterioriaffinityPropagation, metric='cosine', **args):
        self.clustering_parameters = args
        self.algorithm = algorithm
        self.clustering = Clustering(self.algorithm(**self.clustering_parameters))
        self.metric = metric

    def compute_scores(
            self, embs_list: List[np.array],
            timeseries_type='consecutive', k=1, change_metric='apd', time_labels: Union[np.array, List] = None):
        """
        Perform a-posteriori affinity propagation clustering and compute
        semantic change between prototype embeddings across time periods.

        Args:
            embs_list (List[np.array]): Embeddings for one target word, where
                each list element contains the embeddings for one time period.
            timeseries_type (str): Time-series comparison mode as used by
                ``languagechange.models.change.timeseries``.
            k (int): Window size for moving-average comparisons.
            change_metric (str): Change metric to use, for example ``"apd"``.
            time_labels (Union[np.array, List], optional): Labels for the
                x-axis of the resulting time series.

        Returns:
            tuple: ``(labels, prot_embs, change_scores)`` where ``labels`` is
            the cluster label array for each time period, ``prot_embs`` is the
            list of prototype embedding matrices per period, and
            ``change_scores`` is a ``TimeSeries`` object describing the degree
            of change across periods.
        """
        self.clustering = Clustering(self.algorithm(**self.clustering_parameters))
        self.clustering.get_cluster_results(embs_list)
        all_labels = self.clustering.labels
        labels = []

        i = 0
        for embs in embs_list:
            labels.append(all_labels[i:i+embs.shape[0]])
            i += embs.shape[0]

        # Compute the centroids of each cluster (the prototype embeddings)
        prot_embs = []
        for i, embs in enumerate(embs_list):
            prot_embs.append(np.array([embs[labels[i] == label].mean(axis=0) for label in np.unique(labels[i])]))

        # Get the change scores between prototype embeddings
        change_scores = TimeSeries(embeddings_or_cluster_labels=prot_embs, change_metric=change_metric,
                                   timeseries_type=timeseries_type, k=k, time_labels=time_labels)

        return labels, prot_embs, change_scores
