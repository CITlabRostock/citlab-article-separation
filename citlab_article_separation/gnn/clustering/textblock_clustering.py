import logging
import configparser
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import dbscan
from citlab_article_separation.gnn.clustering.dbscan import DBScanRelation


class TextblockClustering(object):
    """
    class for textblock clustering

    Usage::

        tb2ni = TB2NI(config)
        …
        tb2ni.set_confs(confListList, symmFct=scipy.stats.gmean)
        tb2ni.calc(method='greedy')
        print(tb2ni.cntClasses, tb2ni.cntNoise)
        print(tb2ni.tbLabels, tb2ni.tbClasses)
        print(tb2ni.relLLH)

    """
    #: configparser.ConfigParser:   parameter container
    params = None
    #: list of int: textblock labels
    tb_labels = None
    #: list of list of int: textblock classes
    tb_classes = None
    #: int:    counts classes
    num_classes = 0
    #: int: counts noise classes (see DBSCAN algorithm)
    num_noise = 0
    #: float: special relative -loglikelihood value (for internal comparison only)
    rel_LLH = 0.0

    def __init__(self, config_path):
        """
        standard constructor

        Args:
            config_path:  path to config file containing all params in section per method
        """
        self.params = configparser.ConfigParser()
        self.params.read(config_path)
        self._dbscanner = None
        self._conf_mat = None
        self._mat_dim = None
        self._dist_mat = None
        self._cond_dist_list = None
        self._delta_mat = None

    def set_confs(self, confs, symmetry_fn=None):
        """
        sets confidence values and symmetrization function

        Args:
            confs (list of list of floats): confidence values from (0,1)
            symmetry_fn (numpy-like matrix function):   function to make confidences symmetric

        Note:
            symmetry_fn will be applied to confidences, hence geometric is preferred over arithmetic mean.
            But suitable is any (numpy or scipy) function element-wisely applicable to 2D-arrays (e.g. numpy.max/min).
        """
        self._conf_mat = np.array(confs)
        self._mat_dim = self._conf_mat.shape[0]
        if symmetry_fn:
            self._make_symmetric(symmetry_fn)
        self._dist_mat = -np.log(self._conf_mat)
        #   linkage
        self._cond_dist_list = []
        for i in range(self._mat_dim):
            for j in range(i + 1, self._mat_dim):
                self._cond_dist_list.append(self._dist_mat[i, j])
        logging.debug(f'… generated condensed list of length {len(self._cond_dist_list)}')
        #   greedy
        self._delta_mat = np.array(list(map(lambda p: np.log(p / (1 - p)), self._conf_mat)))
        #   Hauptdiagonalen
        for i in range(self._mat_dim):
            self._conf_mat[i, i] = 0.0
            self._delta_mat[i, i] = -np.math.inf

    def _make_symmetric(self, symmetry_fn):
        mat = self._conf_mat
        mat_transpose = self._conf_mat.transpose()
        temp_mat = np.stack([mat, mat_transpose], axis=-1)
        self._conf_mat = symmetry_fn(temp_mat, axis=-1)

    def calc(self, method):
        """
        run calculation of clusters

        Args:
            method (str): method name

        Note:
            currently implemented methods: 'dbscan', 'linkage', 'greedy', 'dbscanStd'
        """
        self.tb_labels = None
        self.tb_classes = None
        if getattr(self, f'_{method}', None):
            fctn = getattr(self, f'_{method}', None)
            logging.info(f'performing clustering with method "{method}"')
            fctn()
        else:
            raise NotImplementedError(f'cannot find clustering method "_{method}"!')
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

    def _dbscanStd(self):
        minSamples = self.params.getint('dbscan', 'min_samples', fallback=1)
        epsilon = self.params.getfloat('dbscan', 'epsilon', fallback=0.5)
        (_, self.tb_labels) = dbscan(self._dist_mat, metric='precomputed', min_samples=minSamples,
                                     eps=epsilon)
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _greedy(self):
        iter_max = self.params.getint('greedy', 'maxIteration', fallback=None)
        self.tb_labels = np.array(range(self._mat_dim), dtype=int)
        self._labels2classes()
        self._calcMat = self._delta_mat.copy()
        iter_count = iter_max
        while iter_count > 0:
            iter_count -= 1
            #  stärkste Kante
            (i, j) = np.unravel_index(np.argmax(self._calcMat), self._conf_mat.shape)
            if self._calcMat[i, j] > 0:
                logging.debug(f'{i}–{j} mit {self._calcMat[i, j]} …')
                self._greedy_step(i, j)
                self._classes2labels()
                self._calc_relative_LLH()
                logging.debug(f'… LLH = {self.rel_LLH}')
                logging.debug(f'{self.tb_labels}')
                logging.debug(f'{self.tb_classes}')
            else:
                logging.info(f'… after {iter_max - iter_count} iterations')
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
            if idx != cls0:
                self._calcMat[idx, cls0] += self._calcMat[idx, cls1]
                self._calcMat[cls0, idx] = self._calcMat[idx, cls0]

        for idx in range(self._mat_dim):
            self._calcMat[idx, cls1] = -np.math.inf
            self._calcMat[cls1, idx] = self._calcMat[idx, cls1]

    def _linkage(self):
        method = self.params.get('linkage', 'method', fallback='centroid')
        criterion = self.params.get('linkage', 'criterion', fallback='distance')
        t = self.params.getfloat('linkage', 't', fallback=-1.0)
        linkageResult = linkage(np.array(self._cond_dist_list, dtype=float), method=method)

        if t < 0:
            hierarchicalDistances = linkageResult[:, 2]
            distanceMean = float(np.mean(hierarchicalDistances))
            distanceMedian = float(np.median(hierarchicalDistances))
            t = 1 / 2 * (distanceMean + distanceMedian)

        self.tb_labels = fcluster(linkageResult, t=t, criterion=criterion)
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])

    def _dbscan(self):
        if not self._dbscanner:
            min_neighbors_for_cluster = self.params.getint('dbscan', 'min_neighbors_for_cluster', fallback=1)
            confidence_threshold = self.params.getfloat('dbscan', 'confidence_threshold', fallback=0.5)
            cluster_agreement_threshold = self.params.getfloat('dbscan', 'cluster_agreement_threshold', fallback=0.5)
            assign_noise_clusters = self.params.getboolean('dbscan', 'assign_noise_clusters', fallback=True)
            self._dbscanner = DBScanRelation(min_neighbors_for_cluster=min_neighbors_for_cluster,
                                             confidence_threshold=confidence_threshold,
                                             cluster_agreement_threshold=cluster_agreement_threshold,
                                             assign_noise_clusters=assign_noise_clusters)

        self.tb_labels = self._dbscanner.cluster_relations(self._mat_dim, self._conf_mat)
        self._labels2classes()
        self.num_classes = len(self.tb_classes)
        self.num_noise = len([label for label in self.tb_labels if label == -1])
