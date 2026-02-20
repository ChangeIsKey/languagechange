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
from languagechange.cache import CacheManager

def generate_cache_key(embeddings):
    """
    Generate a unique cache key based on the input data.
    """
    try:
        serialized = pickle.dumps(embeddings)
        return hashlib.sha256(serialized).hexdigest()
    except Exception as e:
        raise ValueError(f"Invalid input: {e}")

class Visualizer():
    def __init__(self, cache_dir="~/.cache/languagechange/reduced_embeddings"):
        self.cache_mgr = CacheManager(cache_dir)

    def plot_usage_embeddings(self, embeddings,
                              usages=None,
                              cluster_labels=None,
                              one_plot=False,
                              counts_per_time_period=None,
                              time_labels=None,
                              target=None,
                              ncols=3,
                              plot_w=30,
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
            treated as "No cluster").
        
            Time periods can be provided explicitly via 'counts_per_time_period', or inferred from 'usages' by grouping 
            by 'usage.time' (after sorting by time).
    
            Args:
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
                one_plot (bool, default=False):
                    If True, plot all embeddings in a single subplot and ignore time period splitting.
                counts_per_time_period (list[int], default=None):
                    The amount of embeddings to use for each time period or domain. If provided, embeddings will be 
                    split into a list of embeddings according to the numbers specified. Required if 'one_plot=False' 
                    and 'usages' is not provided.
                time_labels (list, default=None): custom labels to use for the titles of subplots. By default, the 
                    years of the usages are used, if possible.
                target (str, default=None):
                    Optional target word, added to the title of the subplot(s).
                ncols (int, default=3):
                    Number of subplot columns when plotting multiple time periods. Ignored if 'one_plot=True'.
                plot_w (float, default=30):
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
        indices = None
        # Initialize from a list of embeddings belonging to different time periods
        if isinstance(embeddings, list) and all(isinstance(e, np.ndarray) for e in embeddings):
            indices = np.cumsum([0] + [len(e) for e in embeddings])
            embeddings = np.concatenate(embeddings)
        elif not isinstance(embeddings, np.ndarray):
            raise TypeError("'embeddings' must be either a numpy array of embeddings or a list of such.")
    
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
        
        if one_plot:
            ncols = 1
            n_time_periods = 1
            indices = [0, len(embeddings)]
        else:
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
                else:
                    logging.error("'embeddings_per_time_period' was not provided, and could not be automatically derived from usages.")
                    raise ValueError
            n_time_periods = len(indices) - 1

            if time_labels is not None and len(time_labels) != n_time_periods:
                logging.info(time_labels)
                logging.info("The length of 'time_labels' does not equal the amount of time periods derived from embeddings.")
                logging.info("Setting 'time_labels' to None.")
                time_labels = None
    
        kwargs["perplexity"] = min(kwargs.get("perplexity", 30), len(embeddings) - 1)
        cache_key = generate_cache_key({"emb": embeddings, **kwargs})
        cache_path = os.path.join(self.cache_mgr.cache_dir, f"{cache_key}.npy")
        if os.path.exists(cache_path):
            try:
                logging.info(f"Loading cached embeddings from {cache_path}")
                reduced_embs = np.load(cache_path, allow_pickle=True)
            except Exception as e:
                logging.error(f"Cache loading failed: {str(e)}, deleting corrupted cache file...")
                os.remove(cache_path)
        else:
            logging.info("Projecting embeddings to two dimensions using t-SNE.")
            reduced_embs = TSNE(n_components=2, learning_rate=learning_rate, init=init, **kwargs).fit_transform(embeddings)
            with self.cache_mgr.atomic_write(cache_path) as temp_path:
                np.save(temp_path, reduced_embs)
                logging.info("Done. Reduced embeddings were saved.")
    
        nrows = math.ceil(n_time_periods/ncols)
        fig, axs = plt.subplots(ncols=ncols, nrows=nrows, sharex = True, sharey = True)
    
        plot_h = plot_w * math.ceil(n_time_periods/ncols) / ncols
    
        fig.set_figwidth(plot_w)
        fig.set_figheight(plot_h)
        marker_size = plot_w / ncols
    
        if cluster_labels is not None:
            unique_clusters = set(cluster_labels).difference({-1,"-1"})
            label_index = {-1: -1, "-1": -1} | {l: i for i, l in enumerate(unique_clusters)}
            n_classes = len(unique_clusters)
            # Generate a colormap with colors that are distinguishable from each other
            hues = np.linspace(0, 1, n_classes, endpoint=False)
            saturations = np.full(n_classes, 0.5)
            values = np.tile(np.linspace(0.5,1,3),n_classes//3+1)[:n_classes]
            hsv = np.stack([hues, saturations, values], axis=1)
            including_grey = np.vstack(([(0.7,0.7,0.7)], mpl.colors.hsv_to_rgb(hsv)))
            cmap = mpl.colors.ListedColormap(including_grey)
        
        for t in range(n_time_periods):
            if ncols == 1 and nrows == 1:
                ax = axs
            elif ncols < n_time_periods and ncols > 1:
                ax = axs[t // ncols][t % ncols]
            else:
                ax = axs[t]
    
            period_reduced_embs = reduced_embs[indices[t]:indices[t+1]]
            
            if cluster_labels is not None:
                period_cluster_labels = cluster_labels[indices[t]:indices[t+1]]
        
                for label in sorted(np.unique(period_cluster_labels)):
                    cluster_embs = period_reduced_embs[np.where(period_cluster_labels == label)]
                    x = cluster_embs[:,0]
                    y = cluster_embs[:,1]
                    legend_label = "No cluster" if label == -1 or label == "-1" else f"Cluster {int(label)}"
                    ax.scatter(x,y, c=cmap(label_index[label]+1), s=marker_size, label=legend_label)
    
                if plot_cluster_labels:
                    ax.legend()
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