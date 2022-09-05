import os
import re
import time
import numpy as np
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram, cophenet, cut_tree
from sklearn.cluster import dbscan
from scipy.stats import gmean
import kneed
# import c_index
from sklearn.metrics import silhouette_score
from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation.gnn.clustering.clustering_validity import calinkski_harabasz_score, c_index_score, \
    connectivity_score
from citlab_article_separation.gnn.clustering.dbscan import DBScanRelation
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")


class TextblockClustering(object):
    """
    class for textblock clustering

    Usage::

        tb_clustering = TextblockClustering(flags)
        …
        tb_clustering.set_confs(confs, symmetry_fn=scipy.stats.gmean)
        tb_clustering.calc(method='greedy')
        print(tb_clustering.num_classes, tb_clustering.num_noise)
        print(tb_clustering.tb_labels, tb_clustering.tb_classes)
        print(tb_clustering.rel_LLH)

    """

    def __init__(self, flags):
        self.clustering_params = dict()
        self._flags = flags

        # Default params which are method dependant
        # [dbscan]
        self.clustering_params["min_neighbors_for_cluster"] = 1
        self.clustering_params["confidence_threshold"] = 0.5
        self.clustering_params["cluster_agreement_threshold"] = 0.5
        self.clustering_params["assign_noise_clusters"] = True
        # [linkage]
        self.clustering_params["method"] = "average"
        self.clustering_params["criterion"] = "maxclust"
        self.clustering_params["t"] = -1.0
        self.clustering_params["max_clusters"] = 100
        # [greedy]
        self.clustering_params["max_iteration"] = 1000
        # [dbscan_std]
        self.clustering_params["epsilon"] = 0.5
        self.clustering_params["min_samples"] = 1

        # Updating of the default params if provided via flags as a dict
        for i in self._flags.clustering_params:
            if i not in self.clustering_params:
                logger.critical(f"Given input_params-key '{i}' is not used by class 'TextblockClustering'!")
        self.clustering_params.update(flags.clustering_params)

        # Public params to access clustering results
        self.tb_labels = None  # list of int: textblock labels
        self.tb_classes = None  # list of list of int: textblock classes
        self.num_classes = 0  # int: counts classes
        self.num_noise = 0  # int: counts noise classes (see DBSCAN algorithm)
        self.rel_LLH = 0.0  # float: special relative-loglikelihood value (for internal comparison only)

        # Private params for internal computations
        self._conf_mat = None
        self._mat_dim = None
        self._dist_mat = None
        self._cond_dists = None
        self._delta_mat = None
        self._dbscanner = None

        # Debug
        self._page_path = None
        self._save_dir = None
        self._debug = False

    def set_debug(self, page_path, save_dir):
        self._page_path = page_path
        self._save_dir = save_dir
        if self._page_path and self._save_dir:
            self._debug = True

    def print_params(self):
        logger.info("CLUSTERING:")
        sorted_dict = sorted(self.clustering_params.items(), key=lambda kv: kv[0])
        for a in sorted_dict:
            logger.info(f"  {a[0]}: {a[1]}")

    def get_info(self, method):
        """Returns an info string with the most important paramters of the given method"""
        info_string = None
        if getattr(self, f'_{method}', None):
            if method == 'dbscan':
                info_string = f'dbscan_conf{self.clustering_params["confidence_threshold"]}_' \
                              f'cluster{self.clustering_params["cluster_agreement_threshold"]}'
            elif method == 'dbscan_std':
                info_string = f'dbscan_std_eps{self.clustering_params["epsilon"]}_' \
                              f'samples{self.clustering_params["min_samples"]}'
            elif method == 'linkage':
                info_string = f'linkage_{self.clustering_params["method"]}_{self.clustering_params["criterion"]}_' \
                              f'{self.clustering_params["t"]}'
            elif method == 'greedy':
                info_string = f'greedy_iter{self.clustering_params["max_iteration"]}'
        return info_string

    def set_confs(self, confs, symmetry_fn=gmean):
        """
        Sets confidence values and symmetrization function.

        Note that since the `symmetry_fn` will be applied to confidences, the geometric mean is preferred over
        the arithmetic mean. But suitable is any function element-wise applicable to 2D-arrays.

        :param confs: confidence array with values from (0,1)
        :param symmetry_fn: array-like function to make confidences symmetric
        :return: None
        """
        self._conf_mat = np.array(confs)
        self._mat_dim = self._conf_mat.shape[0]
        # Substitute confidences of 0.0 and 1.0 with next bigger/smaller float (to prevent zero divides)
        self._smooth_confs()
        # Make confidence matrix symmetric (if not already is)
        if symmetry_fn:
            self._make_symmetric(symmetry_fn)
        # Convert to (pseudo) distances
        self._dist_mat = -np.log(self._conf_mat)
        # Distances for same elements should be 0
        np.fill_diagonal(self._dist_mat, 0.0)
        #   linkage (condensed distance array)
        cond_indices = np.triu_indices_from(self._dist_mat, k=1)
        self._cond_dists = self._dist_mat[cond_indices]
        #   greedy
        self._delta_mat = np.array(list(map(lambda p: np.log(p / (1 - p)), self._conf_mat)))
        np.fill_diagonal(self._delta_mat, -np.math.inf)

    def _make_symmetric(self, symmetry_fn):
        mat = self._conf_mat
        mat_transpose = self._conf_mat.transpose()
        temp_mat = np.stack([mat, mat_transpose], axis=-1)
        self._conf_mat = symmetry_fn(temp_mat, axis=-1)

    def _smooth_confs(self):
        dtype = self._conf_mat.dtype
        min_val = np.nextafter(0, 1, dtype=dtype)
        max_val = np.nextafter(1, 0, dtype=dtype)
        self._conf_mat[self._conf_mat == 0.0] = min_val
        self._conf_mat[self._conf_mat == 1.0] = max_val

    def calc(self, method):
        """
        Run calculation of clusters.

        Currently implemented methods: 'dbscan', 'linkage', 'greedy', 'dbscan_std'

        :param method: method name (str)
        :return: None
        """
        self.tb_labels = None
        self.tb_classes = None

        if self._mat_dim == 2:  # we have exactly two text regions
            logger.info(f'No clustering performed for two text regions. Decision based on confidence '
                         f'threshold ({self._conf_mat[0, 1]} >= {self.clustering_params["confidence_threshold"]}).')
            self.tb_labels = [1, 1] if self._conf_mat[0, 1] >= self.clustering_params["confidence_threshold"] else [1, 2]
        else:  # atleast three text regions
            if getattr(self, f'_{method}', None):
                fctn = getattr(self, f'_{method}', None)
                logger.info(f'Performing clustering with method "{method}"')
                fctn()
            else:
                raise NotImplementedError(f'Cannot find clustering method "_{method}"!')
        self._calc_relative_LLH()

    def _labels2classes(self):
        class_dict = {}
        for (tb, cls) in enumerate(self.tb_labels):
            if class_dict.get(cls, None):
                c = class_dict.get(cls, None)
                c.append(tb)
            else:
                class_dict[cls] = [tb]
        self.tb_classes = list(map(sorted, class_dict.values()))

    def _classes2labels(self):
        self.tb_labels = np.empty(self._mat_dim, dtype=int)
        self.tb_labels.fill(-1)
        for (idx, cls) in enumerate(self.tb_classes):
            for tb in cls:
                self.tb_labels[tb] = idx

    def _calc_relative_LLH(self):
        self.rel_LLH = 0.0
        for idx0 in range(self._mat_dim):
            if self.tb_labels[idx0] >= 0:
                for idx1 in range(idx0):
                    if self.tb_labels[idx0] == self.tb_labels[idx1]:
                        delta_LLH = (self._delta_mat[idx0, idx1] + self._delta_mat[idx1, idx0]) / 2
                        self.rel_LLH += delta_LLH

    def _dbscan_std(self):
        (_, self.tb_labels) = dbscan(self._dist_mat,
                                     metric='precomputed',
                                     min_samples=self.clustering_params["min_samples"],
                                     eps=self.clustering_params["epsilon"])
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _greedy(self):
        self.tb_labels = np.array(range(self._mat_dim), dtype=int)
        self._labels2classes()
        self._calcMat = self._delta_mat.copy()
        iter_count = self.clustering_params["max_iteration"]
        while iter_count > 0:
            iter_count -= 1
            #  stärkste Kante
            (i, j) = np.unravel_index(np.argmax(self._calcMat), self._conf_mat.shape)
            if self._calcMat[i, j] > 0:
                logger.debug(f'{i}–{j} mit {self._calcMat[i, j]} …')
                self._greedy_step(i, j)
                self._classes2labels()
                self._calc_relative_LLH()
                logger.debug(f'… LLH = {self.rel_LLH}')
                logger.debug(f'{self.tb_labels}')
                logger.debug(f'{self.tb_classes}')
            else:
                logger.debug(f'… after {self.clustering_params["max_iteration"] - iter_count} iterations')
                break

        self.tb_classes = [cls for cls in self.tb_classes if len(cls) > 0]
        self.num_classes = len(self.tb_classes)
        self._classes2labels()
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _greedy_step(self, cls0, cls1):
        self.tb_classes[cls0].extend(self.tb_classes[cls1])
        self.tb_classes[cls0] = sorted(self.tb_classes[cls0])
        self.tb_classes[cls1] = []

        for idx in range(self._mat_dim):
            if idx != cls0 and idx != cls1:
                self._calcMat[idx, cls0] += self._calcMat[idx, cls1]
                self._calcMat[cls0, idx] = self._calcMat[idx, cls0]

        for idx in range(self._mat_dim):
            self._calcMat[idx, cls1] = -np.math.inf
            self._calcMat[cls1, idx] = self._calcMat[idx, cls1]

    def _linkage(self):
        linkage_res = linkage(self._cond_dists, method=self.clustering_params["method"])
        if self.clustering_params["t"] == -1:
            logger.info(f"linkage with distance thresholding.")
            hierarchical_distances = linkage_res[:, 2]
            distance_mean = float(np.mean(hierarchical_distances))
            distance_median = float(np.median(hierarchical_distances))
            t = 1 / 2 * (distance_mean + distance_median)
            self.tb_labels = fcluster(linkage_res, t=t, criterion=self.clustering_params["criterion"])
        else:
            num_clusters, labels = self._validate_clusters(linkage_res)
            if self._debug:
                self._debug_linkage(linkage_res, num_clusters)
            # self.tb_labels = fcluster(linkage_res, t=t, criterion=self.clustering_params["criterion"])
            # self.tb_labels = fcluster(linkage_res, t=num_clusters, criterion=self.clustering_params["criterion"])
            self.tb_labels = labels
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _validate_clusters(self, linkage_res):
        timer = time.time()
        # Setup clustering validity indices
        s_scores = []  # silhouette scores
        # ch_scores = []  # calinkski harabasz scores
        c_scores = []  # c-index
        ssw_scores = []  # sum-squares-within clusters
        ssb_scores = []  # sum-squares-between clusters
        sst_scores = []  # sum-squares-total clusters
        # con_scores = []  # connectivity scores

        # Validate possible clusterings
        max_clusters = min(self._mat_dim, self.clustering_params["max_clusters"])  # try at most N clusters
        tree = cut_tree(linkage_res)  # all clusterings based on linkage result
        tree = np.transpose(tree[:, ::-1])[:max_clusters, :]
        labels_list = tree.tolist()
        cluster_nums = list(range(1, len(labels_list) + 1))
        for cluster_num, labels in zip(cluster_nums, labels_list):
            if cluster_num == 1:  # return single cluster or skip
                cond_indices = np.triu_indices_from(self._conf_mat, k=1)
                cond_confs = self._conf_mat[cond_indices]  # upper triangular matrix
                if np.all(cond_confs >= self.clustering_params["confidence_threshold"]):
                    # all confidences above threshold
                    return 1, labels_list[0]
                else:
                    # no validity indices for single cluster
                    continue
            try:
                s = silhouette_score(self._dist_mat, labels, metric='precomputed')
            except ValueError:  # not defined if num_labels == num_samples
                s = 0.0
            s_scores.append(s)
            # ssw, ssb, sst, ch = calinkski_harabasz_score(self._dist_mat, labels, is_squared=False)
            # ch_scores.append(ch)
            # ssw_scores.append(ssw)
            # ssb_scores.append(ssb)
            # sst_scores.append(sst)
            # con = connectivity_score(self._dist_mat, labels, num_neighbors=5)
            # con_scores.append(con)
            # c = c_index_score(self._dist_mat, labels, is_squared=False)
            # c_scores.append(c)
            # c1 = c_index.calc_cindex_clusterSim_implementation(c_index.pdist_array(self._dist_mat), labels)
            # c1_scores.append(c1)
            # c2 = c_index.calc_cindex_nbclust_implementation(c_index.pdist_array(self._dist_mat), labels)
            # c2_scores.append(c2)

        # Elbow method for validity indices
        cluster_by_elbow = dict()
        cluster_num = self._elbow_method(cluster_nums[1:], s_scores, "silhouette", 'concave', 'increasing')
        cluster_by_elbow["silhouette"] = cluster_num
        # elbow = self._elbow_method(cluster_nums[1:], con_scores, "connectivity", 'convex', 'increasing')
        # cluster_by_elbow["connectivity"] = elbow
        # cluster_num = self._elbow_method(cluster_nums[1:], c_scores, "c_index", 'convex', 'decreasing')
        # cluster_by_elbow["c_index"] = cluster_num
        # elbow = self._elbow_method(cluster_nums[1:], c1_scores, "c_index_sim", 'convex', 'decreasing')
        # cluster_by_elbow["c1_index"] = elbow
        # elbow = self._elbow_method(cluster_nums[1:], c2_scores, "c_index_nbClust", 'convex', 'decreasing')
        # cluster_by_elbow["c2_index"] = elbow
        # elbow = self._elbow_method(cluster_nums[1:], ch_scores, "calinkski_harabasz", 'convex', 'increasing')
        # cluster_by_elbow["ch_score"] = elbow
        # cluster_num = self._elbow_method(cluster_nums[1:], ssw_scores, "ssw", 'convex', 'decreasing')
        # cluster_by_elbow["ssw"] = cluster_num
        # elbow = self._elbow_method(cluster_nums[1:], ssb_scores, "ssb", 'concave', 'increasing')
        # cluster_by_elbow["ssb"] = elbow
        # ssw_scores[-1] = ssw_scores[-2]  # set last ssw score (which is 0) to previous value to prevent zero-divide
        # online = False to take first knee in hartigan
        # cluster_num = self._elbow_method(cluster_nums[1:], np.log(np.array(ssb_scores) / np.array(ssw_scores)),
        #                                  "hartigan", 'concave', 'increasing', online=False)
        # cluster_by_elbow["hartigan"] = cluster_num
        # cluster merge distances
        last_merges = linkage_res[-int(max_clusters):, 2]
        last_merges = np.concatenate(([0.0], last_merges), axis=-1)
        idxs = np.arange(1, len(last_merges) + 1, dtype=np.int32)
        cluster_num = self._elbow_method(idxs, last_merges[::-1], "merge_distance", "convex", "decreasing")
        cluster_by_elbow["merge"] = cluster_num
        logger.debug(f"Time for cluster validity check = {time.time() - timer}")

        logger.debug(f"Cluster suggestion by elbow: {cluster_by_elbow}")
        max_silhouette_index = np.argmax(s_scores) + 2  # first silhouette score is at two clusters
        logger.debug(f"Max silhouette score at {max_silhouette_index} clusters")
        if self.clustering_params["t"] == "silhouette":
            num_clusters = max_silhouette_index
        else:
            try:
                num_clusters = cluster_by_elbow[self.clustering_params["t"]]
            except KeyError:
                logger.error(f'Clustering param t = {self.clustering_params["t"]} not in validity indices. '
                             f'Defaulting to num_clusters = 1')
                num_clusters = 1
        num_clusters = num_clusters if num_clusters is not None else 1
        return num_clusters, labels_list[num_clusters - 1]

    def _elbow_method(self, x, y, name, curve, direction, online=True):
        # see https://towardsdatascience.com/cheat-sheet-to-implementing-7-methods-for-selecting-optimal-number-of-clusters-in-python-898241e1d6ad
        # see https://github.com/arvkevi/kneed for implementation
        # for sensitivity in range(1, 0, -1):  # start conservative and work down
        #     kneedle = kneed.KneeLocator(x, y, curve=curve, direction=direction, S=sensitivity, online=True)
        #     elbow = kneedle.elbow
        #     if elbow is not None:  # break if elbow was found
        #         break
        sens = 1.0
        kneedle = kneed.KneeLocator(x, y, curve=curve, direction=direction, S=sens, online=online)
        # kneedle = kneed.KneeLocator(x, y, curve=curve, direction=direction, S=sens, online=True,
        #                             interp_method="polynomial", polynomial_degree=7)
        # try:  # take second elbow if available
        #     elbow = list(kneedle.all_elbows)[1]
        # except IndexError:
        #     elbow = kneedle.elbow
        elbow = kneedle.elbow
        logger.debug(f"{name} elbows (s = {sens}): {sorted(kneedle.all_elbows)}")
        if self._debug:
            self._debug_elbows(x, y, kneedle.all_elbows, name)
        return elbow

    def _debug_elbows(self, x, y, elbows, name):
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.set_title(f"{name} - {self.get_info(self._flags.clustering_method)}")
        ax.plot(x, y, marker='o', label=f"{name}")
        growth = np.diff(y, 1)
        acceleration = np.diff(y, 2)
        ax.plot(x[1:], growth, marker='o', label="growth")
        ax.plot(x[2:], acceleration, marker='o', label="acceleration")
        ax.xaxis.set_ticks(np.arange(np.min(x), np.max(x) + 1))
        ax.axhline(y=0, color='black', linestyle='-')  # x-axis at y = 0

        label = "elbow"
        for elbow in elbows:
            ax.axvline(elbow, ymin=0.0, ymax=1.0, linestyle=":", color="black", label=label)  # elbow indicators
            ax.text(elbow, 0.5, str(elbow), transform=ax.get_xaxis_text1_transform(0)[0])  # elbow number
            label = ""  # only set label once for elbows
        ax.legend()
        # Save image
        page_path = os.path.relpath(self._page_path)
        save_name = re.sub(r'\.xml$', f'_elbow_{name}.jpg', os.path.basename(page_path))
        page_dir = os.path.dirname(page_path)
        info = self.get_info(self._flags.clustering_method)
        if self._flags.mask_heading_separated_confs:
            info += "_maskHead"
        if self._flags.mask_horizontally_separated_confs:
            info += "_maskSep"
        save_dir = os.path.join(self._save_dir, page_dir, info)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        logger.debug(f"Saved {name} elbow image '{save_path}'")
        plt.close(plt.gcf())

    def _debug_linkage(self, linkage_res, num_clusters):
        # https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
        page = Page(self._page_path)
        logger.debug("Page:")
        logger.debug(f"  Number of text_regions = {len(page.get_text_regions())}")
        logger.debug(f"  Number of articles = {len(page.get_article_region_dicts()[0].keys())}")

        hierarchical_distances = linkage_res[:, 2]
        coph_corr_coeff, coph_dists = cophenet(linkage_res, self._cond_dists)
        ha_dist_mean = float(np.mean(hierarchical_distances))
        ha_dist_median = float(np.median(hierarchical_distances))
        ha_dist_max = float(np.max(hierarchical_distances))
        gnn_dist_mean = float(np.mean(self._dist_mat))
        gnn_dist_median = float(np.median(self._dist_mat))
        gnn_dist_max = float(np.max(self._dist_mat))
        coph_dist_mean = float(np.mean(coph_dists))
        coph_dist_median = float(np.median(coph_dists))
        coph_dist_max = float(np.max(coph_dists))
        logger.debug("Distances:")
        logger.debug(f"  linkage_dists_max = {ha_dist_max:.3f}")
        logger.debug(f"  linkage_dists_mean = {ha_dist_mean:.3f}")
        logger.debug(f"  linkage_dists_median = {ha_dist_median:.3f}")
        logger.debug(f"  gnn_dists_max = {gnn_dist_max:.3f}")
        logger.debug(f"  gnn_dists_mean = {gnn_dist_mean:.3f}")
        logger.debug(f"  gnn_dists_median = {gnn_dist_median:.3f}")
        logger.debug(f"  coph_dists_max = {coph_dist_max:.3f}")
        logger.debug(f"  coph_dists_mean = {coph_dist_mean:.3f}")
        logger.debug(f"  coph_dists_median = {coph_dist_median:.3f}")
        t_old = 1 / 2 * (ha_dist_mean + ha_dist_median)
        t = 2 * ha_dist_mean
        logger.debug(f"  Threshold_old = {t_old:.3f}")
        logger.debug(f"  Threshold = {t:.3f}")
        logger.debug(f"  Cophenetic Correlation Coefficient = {coph_corr_coeff}")

        logger.debug(f"Final number of clusters = {num_clusters}")
        dist_maxclust_low = linkage_res[-num_clusters, 2]
        dist_maxclust_high = linkage_res[-(num_clusters - 1), 2]
        logger.debug(f"  corresponding merge dist range: [{dist_maxclust_low:.3f}, {dist_maxclust_high:.3f}]")

        # Create dendogram figure
        fig, ax = plt.subplots(1, 1, figsize=(15, 15))
        ax.set_title(f"Dendogram, {self.get_info(self._flags.clustering_method)}")
        color_thresh = (dist_maxclust_low + dist_maxclust_high) / 2
        dendrogram(linkage_res, ax=ax, color_threshold=color_thresh,
                   count_sort=True, show_contracted=True, show_leaf_counts=True, leaf_rotation=45)
        # stepsize = 0.1
        # start, end = ax.get_ylim()
        # ax.yaxis.set_ticks(np.arange(start, end, stepsize))

        plt.axhline(y=color_thresh, color='r', linestyle=':')
        ax.annotate(f'final', xy=(1, color_thresh), xytext=(0, color_thresh))
        plt.axhline(y=ha_dist_mean, color='r', linestyle=':')
        ax.annotate(f'linkage_mean', xy=(1, ha_dist_mean), xytext=(0, ha_dist_mean))
        plt.axhline(y=gnn_dist_mean, color='r', linestyle=':')
        ax.annotate(f'gnn_mean', xy=(1, gnn_dist_mean), xytext=(0, gnn_dist_mean))
        plt.axhline(y=coph_dist_mean, color='r', linestyle=':')
        ax.annotate(f'coph_mean', xy=(1, coph_dist_mean), xytext=(0, coph_dist_mean))
        plt.axhline(y=t_old, color='r', linestyle=':')
        ax.annotate(f't_old', xy=(1, t_old), xytext=(0, t_old))

        # Save image
        page_path = os.path.relpath(self._page_path)
        save_name = re.sub(r'\.xml$', f'_dendogram.jpg', os.path.basename(page_path))
        page_dir = os.path.dirname(page_path)
        info = self.get_info(self._flags.clustering_method)
        if self._flags.mask_heading_separated_confs:
            info += "_maskHead"
        if self._flags.mask_horizontally_separated_confs:
            info += "_maskSep"
        save_dir = os.path.join(self._save_dir, page_dir, info)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, save_name)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        logger.debug(f"Saved dendogram image '{save_path}'")
        plt.close(plt.gcf())

    def _dbscan(self):
        if not self._dbscanner:
            self._dbscanner = DBScanRelation(
                min_neighbors_for_cluster=self.clustering_params["min_neighbors_for_cluster"],
                confidence_threshold=self.clustering_params["confidence_threshold"],
                cluster_agreement_threshold=self.clustering_params["cluster_agreement_threshold"],
                assign_noise_clusters=self.clustering_params["assign_noise_clusters"])

        self.tb_labels = self._dbscanner.cluster_relations(self._mat_dim, self._conf_mat)
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])
