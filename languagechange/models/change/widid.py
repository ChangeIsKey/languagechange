from languagechange.models.meaning.clustering import Clustering, APosterioriaffinityPropagation
from languagechange.models.change.timeseries import TimeSeries
import numpy as np
from typing import List, Union

class WiDiD:
    """
        A class that implements WiDiD (https://github.com/FrancescoPeriti/WiDiD).
    """
    def __init__(self, algorithm = APosterioriaffinityPropagation, metric = 'cosine', **args):
        self.clustering_parameters = args
        self.algorithm = algorithm
        self.clustering = Clustering(self.algorithm(**self.clustering_parameters))
        self.metric = metric

    
    def compute_scores(self, embs_list : List[np.array], timeseries_type='consecutive', k=1, change_metric='apd', time_labels: Union[np.array, List] = None):
        """
            Performs a-posteriori affinity propagation (APP) clustering and computes the semantic change as the APD (or another metric) between the prototype embeddings in clusters of different time periods.
            
            Args: 
                embs_list ([np.array]): a list of embeddings for a target word, where each element is embeddings of one time period.
                timeseries_type (str): the type of timeseries (see usage in languagechange.models.change.timeseries).
                k (int): the window size, if moving average (see usage in languagechange.models.change.timeseries).
                change_metric (str): the change metric (e.g. 'apd') to use (see usage in languagechange.models.change.timeseries).
                time_labels (np.array|list): labels for the x axis of the timeseries (see usage in languagechange.models.change.timeseries).

            Returns:
                labels ([np.array]): the labels for each embedding in each time period.
                prot_embs ([np.array]): a list of matrices encoding the prototype (average) embedding of each cluster in each time period.
                change_scores (TimeSeries): a timeseries (languagechange.models.change.timeseries.TimeSeries) containing the degree of change between the embeddings in different time periods.
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
        change_scores = TimeSeries(embs=prot_embs, change_metric=change_metric, timeseries_type=timeseries_type, k=k, time_labels=time_labels)

        return labels, prot_embs, change_scores