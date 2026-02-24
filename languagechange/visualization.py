from typing import Union
import math
from itertools import groupby
import logging
import os
import pickle
import hashlib
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from IPython.display import Markdown, display
from languagechange.cache import CacheManager
from languagechange.utils import generate_colormap

def generate_cache_key(embeddings):
    """
    Generate a unique cache key based on the input data.
    """
    try:
        serialized = pickle.dumps(embeddings)
        return hashlib.sha256(serialized).hexdigest()
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")

def text_formatting(usage,time_label):
    start, end = usage.offsets
    formatted_text = f'{time_label}:\t' + usage[:start] + '**' + usage[start:end] + '**' + usage[end:]
    return formatted_text

class Visualizer():
    """ 
        Visualize usage embeddings across time and optionally inspect underlying usages.

        This class stores embeddings (optionally grouped by time or domain), usages and optional cluster labels. It can 
        project embeddings to 2D with t-SNE and plot them as scatter plots per time period or domain. It can also 
        display usages grouped by cluster.

        Parameters:
            embeddings (Union[np.ndarray, list[np.ndarray]]):
                Embedding vectors of shape (n_samples, n_features). If a list of arrays is provided, each array
                is treated as a separate time period and concatenated internally.
            usages (list, default=None):
                List of usage objects with a '.time' attribute that can be used for sorting. Used to automatically 
                derive time period boundaries if 'counts_per_time_period' is not provided.
            cluster_labels (np.ndarray or list[np.ndarray]):
                Cluster label per embedding (shape: (n_samples,)). If a list is provided, it must mirror the
                structure of 'embeddings' when embeddings are provided as a list. Label -1 is treated as
                "No cluster".
            counts_per_time_period (list[int], default=None):
                Counts of embeddings per time period. If provided, time period or domain boundaries are computed from
                these counts. If omitted, boundaries may be inferred from 'usages' grouped by 'usage.time'.
            time_labels (list, default=None):
                Labels for time periods or domain names, used in subplot titles (e.g., years). If not provided and 
                `usages` is provided, labels may be inferred from `usage.time`.
            cache_dir (str or Path, default="~/.cache/languagechange/reduced_embeddings"):
                Directory used for caching reduced (2D) embeddings for faster repeated plotting.
    """
    def __init__(self, 
                 embeddings=None,
                 usages=None,
                 cluster_labels=None,
                 counts_per_time_period=None,
                 time_labels=None,
                 cache_dir="~/.cache/languagechange/reduced_embeddings"):
        self.cache_mgr = CacheManager(cache_dir)

        indices = None
        if embeddings is not None:
            # Initialize from a list of embeddings belonging to different time periods
            if isinstance(embeddings, list) and all(isinstance(e, np.ndarray) for e in embeddings):
                indices = np.cumsum([0] + [len(e) for e in embeddings])
                embeddings = np.concatenate(embeddings)
            elif not isinstance(embeddings, np.ndarray):
                logging.error("'embeddings' must be either a numpy array of embeddings or a list of such.")
                raise TypeError

        if usages is not None and len(usages) != len(embeddings):
            logging.error("Number of embeddings does not match the number of usages.")
            raise ValueError
    
        if cluster_labels is not None:
            if isinstance(cluster_labels, list) and all(isinstance(l, np.ndarray) for l in cluster_labels):
                label_indices = np.cumsum([0] + [len(l) for l in cluster_labels])
                if indices is not None and not (label_indices == indices).all():
                    logging.error("'embeddings' and 'cluster_labels' have different amounts of items in each part of the list.")
                    raise ValueError
                cluster_labels = np.concatenate(cluster_labels)
            elif not isinstance(cluster_labels, np.ndarray) and cluster_labels is not None:
                raise TypeError("'cluster_labels' must be either None, a numpy array of labels or a list of such.")
        
            if len(cluster_labels) != len(embeddings):
                logging.error("Number of embeddings does not match the number of cluster labels.")
                raise ValueError
            self.cluster_labels = cluster_labels
        
        # Split the embeddings into a list of embeddings by year
        if counts_per_time_period is not None:
            indices = np.cumsum([0] + counts_per_time_period)
        elif indices is None:
            if usages is not None and all(hasattr(u, "time") for u in usages):
                # Sort usages by time, and reorder embeddings and cluster labels by the same ordering
                new_order, sorted_usages = zip(*sorted(enumerate(usages), key = lambda u : u[1].time))
                usages = list(sorted_usages)
                embeddings = embeddings[list(new_order)]
                cluster_labels = cluster_labels[list(new_order)] if cluster_labels is not None else None

                # Group usages by time and count each group to automatically derive indices
                usages_by_time = groupby(usages, key = lambda u : u.time)
                counts = []
                ts = []
                for t, us in usages_by_time:
                    counts.append(len(list(us)))
                    ts.append(t)
                indices = np.cumsum([0] + counts)

                # Add time labels if they were not already provided
                if time_labels is None:
                    time_labels = ts

        if time_labels is not None and (indices is None or len(time_labels) != len(indices) - 1):
            logging.info(time_labels)
            logging.info("The length of 'time_labels' does not equal the amount of time periods derived from embeddings.")
            logging.info("Setting 'time_labels' to None.")
            time_labels = None

        self.embeddings = embeddings
        self.usages = usages
        self.cluster_labels = cluster_labels
        self.indices = indices
        self.time_labels = time_labels

    def plot_usage_embeddings(self,
                              one_plot=False,
                              counts_per_time_period=None,
                              time_labels=None,
                              target=None,
                              ncols=3,
                              plot_w=15,
                              plot_cluster_labels=True,
                              plot=True,
                              save_f=None,
                              learning_rate="auto",
                              init="random",
                              **kwargs):
        """
            Plot 2D t-SNE projections of usage embeddings, optionally split into time periods or domains.
    
            This function reduces high-dimensional 'embeddings' to 2D using t-SNE and creates one scatter plot per time 
            period (or a single plot if 'one_plot=True'). Points are colored by 'cluster_labels' (with label '-1' 
            treated as 'No cluster').
        
            Time periods can be provided explicitly via 'counts_per_time_period', or inferred from 'usages' by grouping 
            by 'usage.time' (after sorting by time).
    
            Args:
                one_plot (bool, default=False):
                    If True, plot all embeddings in a single subplot and ignore time period splitting.
                counts_per_time_period (list[int], default=None):
                    The amount of embeddings to use for each time period or domain. If provided, embeddings will be 
                    split into a list of embeddings according to the numbers specified. Required if 'one_plot=False' 
                    and 'usages' is not provided.
                time_labels (list, default=None): Labels used in subplot titles for each time period or domain. If 
                    None, falls back to 'self.time_labels' if available.
                target (str, default=None):
                    Optional target word, added to the title of the subplot(s).
                ncols (int, default=3):
                    Number of subplot columns when plotting multiple time periods. Ignored if 'one_plot=True'.
                plot_w (float, default=15):
                    Figure width (in inches). Height is derived automatically.
                plot_cluster_labels (bool, default=True):
                    Whether to display a legend showing cluster labels.
                plot (bool, default=True):
                    If True, display the plot using 'plt.show()'.
                save_f (str or Path, default=None):
                    If provided, save the figure to this file path.
                learning_rate (str or float, default="auto"):
                    Learning rate parameter passed to 'sklearn.manifold.TSNE'.
                init (str or np.ndarray, default="random"):
                    Initialization method passed to 'sklearn.manifold.TSNE'.
                **kwargs:
                    Additional keyword arguments forwarded to 'sklearn.manifold.TSNE'.
        """
        if one_plot:
            ncols = 1
            n_time_periods = 1
            indices = [0, len(self.embeddings)]
        elif counts_per_time_period is not None:
            # Split the embeddings into a list of embeddings by year
            indices = np.cumsum([0] + counts_per_time_period)
        elif self.indices is not None:
            indices = self.indices
        else:
            logging.error("Neither 'usages' nor 'counts_per_time_period' provided: non-synchronic plot is not possible.")
            raise ValueError            

        n_time_periods = len(indices) - 1
        time_labels = time_labels or self.time_labels
    
        kwargs["perplexity"] = min(kwargs.get("perplexity", 30), len(self.embeddings) - 1)
        cache_key = generate_cache_key({"emb": self.embeddings, "lr": learning_rate, "init": init, **kwargs})
        cache_path = os.path.join(self.cache_mgr.cache_dir, f"{cache_key}.npy")
        compute_embeddings = True
        if os.path.exists(cache_path):
            try:
                logging.info(f"Loading cached embeddings from {cache_path}")
                reduced_embs = np.load(cache_path, allow_pickle=True)
                compute_embeddings = False
            except Exception as e:
                logging.error(f"Cache loading failed: {str(e)}, deleting corrupted cache file...")
                os.remove(cache_path)
        if compute_embeddings:
            logging.info("Projecting embeddings to two dimensions using t-SNE.")
            tsne = TSNE(n_components=2, learning_rate=learning_rate, init=init, **kwargs)
            reduced_embs = tsne.fit_transform(self.embeddings)
            with self.cache_mgr.atomic_write(cache_path) as temp_path:
                np.save(temp_path, reduced_embs)
                logging.info("Done. Reduced embeddings were saved.")
    
        nrows = math.ceil(n_time_periods / ncols)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex = True, sharey = True)
    
        plot_h = plot_w * math.ceil(n_time_periods / ncols) / ncols
    
        fig.set_figwidth(plot_w)
        fig.set_figheight(plot_h)
        marker_size = 1.5 * plot_w / ncols
    
        if self.cluster_labels is not None:
            unique_clusters = sorted(set(self.cluster_labels).difference({-1,"-1"}))
            label_index = {-1: -1, "-1": -1} | {l: i for i, l in enumerate(unique_clusters)}
            n_classes = len(unique_clusters)
            # Generate a colormap with colors that are distinguishable from each other
            cmap = generate_colormap(n_classes)
        
        for t in range(n_time_periods):
            if ncols == 1 and nrows == 1:
                ax = axs
            elif ncols < n_time_periods and ncols > 1:
                ax = axs[t // ncols][t % ncols]
            else:
                ax = axs[t]
    
            period_reduced_embs = reduced_embs[indices[t]:indices[t+1]]
            
            if self.cluster_labels is not None:
                period_cluster_labels = self.cluster_labels[indices[t]:indices[t+1]]
        
                for label in sorted(np.unique(period_cluster_labels)):
                    cluster_embs = period_reduced_embs[np.where(period_cluster_labels == label)]
                    x = cluster_embs[:,0]
                    y = cluster_embs[:,1]
                    if label == -1 or label == "-1":
                        legend_label = "No cluster"
                    else:
                        try:
                            legend_label = str(int(label))
                        except ValueError:
                            legend_label = str(label)
                    ax.scatter(x,y, c=cmap(label_index[label]+1), s=marker_size, label=legend_label)
    
                if plot_cluster_labels:
                    ax.legend(title="Clusters")
            else:
                x = period_reduced_embs[:,0]
                y = period_reduced_embs[:,1]
                ax.scatter(x,y, c="blue", s=marker_size)
    
            if target:
                if one_plot or (ncols == 1 and nrows == 1) or time_labels is None:
                    ax.set_title(f"Usages of '{target}'")
                else:
                    ax.set_title(f"Usages of '{target}' ({time_labels[t]})")
    
        if save_f is not None:
            plt.savefig(save_f, dpi=300)
            logging.info(f"Plot saved to {save_f}.")
        if plot:
            plt.show()

    def _randomize_sort_select(self, indices, time_labels, sort=False, randomize=False, max_usages=None):
        if max_usages is None:
            max_usages = len(indices)

        if randomize:
            chosen = list(np.random.choice(np.arange(len(indices)), min(max_usages, len(indices)), replace=False))
            indices = list(np.array(indices)[chosen])
            if time_labels is not None:
                time_labels = list(np.array(time_labels)[chosen])
        else:
            indices = indices[:max_usages]
            if time_labels is not None:
                time_labels = time_labels[:max_usages]

        if sort:
            indices = [i for i, _ in sorted(zip(indices, time_labels), key = lambda i_t : i_t[1])]

        return indices

    def _display_selected_usages(self, indices):
        for i in indices:
            usage = self.usages[i]
            display(Markdown(text_formatting(usage, usage.time)))

    def display_usages(self, sort=False, max_usages = None, randomize = False):
        """
            Display usages, optionally grouped by cluster, with the target word highlighted. If cluster labels exist, 
            usages are displayed cluster by cluster. Otherwise, usages are shown in a single list. Usages can be 
            randomized and/or sorted by time.

            Args:
                sort (bool, default=False): If True, sort displayed usages by time (ascending). When cluster labels 
                    exist, sorting is applied within each cluster.
                max_usages (int, default=None): Maximum number of usages to display per cluster (or total if no 
                    clusters). If None, display all available usages.
                randomize (bool, default=False): If True, randomly sample up to 'max_usages' usages before optional 
                    sorting.
        """
        if self.usages is None:
            logging.error("Cannot display any usages, as 'usages' was not provided when initializing.")
            raise ValueError
        
        if self.cluster_labels is not None:

            label_usage_dict = {}
            for i, label in enumerate(self.cluster_labels):
                if label not in label_usage_dict:
                    label_usage_dict[label] = []
                label_usage_dict[label].append(i)
                
            for c in sorted(label_usage_dict):
                print(f'Cluster {c}:')
                indices = label_usage_dict[c]
                times = [self.usages[i].time for i in indices]
                indices = self._randomize_sort_select(indices, times, sort, randomize, max_usages)
                self._display_selected_usages(indices)
                print("-" * 80)
        else:
            times = [u.time for u in self.usages]
            indices = self._randomize_sort_select(list(range(len(self.usages))), times, sort, randomize, max_usages)
            self._display_selected_usages(indices)