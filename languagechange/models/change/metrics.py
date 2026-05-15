from scipy.spatial.distance import cdist, cosine
from languagechange.models.meaning.clustering import Clustering
import numpy as np
from collections import Counter
from scipy.spatial.distance import jensenshannon
from typing import List, Union

class ChangeModel():

    def __init__(self):
        pass


class BinaryChange(ChangeModel):

    def __init__(self, not_exist_max_count=2, exist_min_count=5):
        """
        Computes binary change by counting cluster occurrences.

        Parameters:
            not_exist_max_count (int, default=2): the maximum amount of examples allowed in a cluster (for one time 
                period) in order for it to count as a non-existent/vanished sense of the word, if computing binary 
                change scores.
            exist_min_count (int, default=5): the minimum amount of examples needed in a cluster (for one time period) 
                in order for it to count as an existing/emerged sense of the word, if computnig binary change scores.
        """
        self.not_exist_max_count = not_exist_max_count
        self.exist_min_count = exist_min_count

    def compute_scores(self, cluster_labels1, cluster_labels2):
        change = False
        label_counts_t1 = Counter(cluster_labels1)
        label_counts_t2 = Counter(cluster_labels2)
        for l in label_counts_t1.keys() | label_counts_t2.keys():
            if change:
                return 1
            c1, c2 = label_counts_t1[l], label_counts_t2[l]
            change = change or ((c1 < self.not_exist_max_count and c2 > self.exist_min_count) or
                                (c2 < self.not_exist_max_count and c1 > self.exist_min_count))
            
        return int(change)


class GradedChange(ChangeModel):

    def __init__(self):
        pass

    def compute_scores(vectors_list):
        pass


class Threshold(BinaryChange):

    def __init__(self):
        pass

    def set_threshold(self, threshold):
        self.threshold = threshold


class AutomaticThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, func = lambda x: np.mean(x)):
        self.threshold = func(scores)


class OptimalThrehold(Threshold):

    def __init__(self):
        pass

    def compute_threshold(self, scores, vrange=np.arange(0.,1.,100), evaluator=None):
        best_score = None
        best_threshold = None

        for v in vrange:
            labels = np.array(scores > v, dtype=int)
            score = evaluator(labels)
            if best_score is None or score > best_score:
                best_score = score
                best_threshold = v

        self.threshold = best_threshold


class APD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return np.mean(cdist(embeddings1, embeddings2, metric=metric))


class PRT(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, metric='cosine'):

        return cosine(embeddings1.mean(axis=0), embeddings2.mean(axis=0))


class JSD(GradedChange):

    def __init__(self):
        pass

    def compute_scores(self, embeddings1, embeddings2, clustering_algorithm, metric='cosine', return_labels=False):
        clustering = Clustering(clustering_algorithm)
        clustering.get_cluster_results(np.concatenate((embeddings1,embeddings2),axis=0))
        labels1 = clustering.labels[:len(embeddings1)]
        labels2 = clustering.labels[len(embeddings1):]
        if return_labels:
            return self.compute_scores_from_labels(labels1, labels2), (labels1, labels2)
        return self.compute_scores_from_labels(labels1, labels2)

    def compute_scores_from_labels(self, labels1, labels2):
        all_labels = np.concatenate((labels1, labels2),axis=0)
        labels = set(all_labels)
        count1 = Counter(labels1)
        count2 = Counter(labels2)
        p,q = [], []
        for l in labels:
            if l in count1:
                p.append(count1[l]/len(all_labels))
            else:
                p.append(0.)
            if l in count2:
                q.append(count2[l]/len(all_labels))
            else:
                q.append(0.)

        return jensenshannon(p, q)