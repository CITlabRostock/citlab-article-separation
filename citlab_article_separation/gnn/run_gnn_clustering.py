import time
import os
import numpy as np
import tensorflow as tf
import traceback
import multiprocessing as mp
from tensorflow.compat.v1 import placeholder as ph
from sklearn.metrics import precision_recall_curve
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt

import citlab_python_util.basic.flags as flags
from citlab_python_util.io.path_util import get_path_from_exportdir, get_page_from_json_path
from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation.gnn.input.input_dataset import InputGNN
from citlab_article_separation.gnn.input.feature_generation import is_aligned_horizontally_separated, \
    is_aligned_heading_separated, discard_text_regions_and_lines, get_separator_aligned_regions
from citlab_article_separation.gnn.clustering.textblock_clustering import TextblockClustering
from citlab_article_separation.gnn.io import plot_graph_clustering_and_page, save_clustering_to_page, \
    save_conf_to_json, build_thresholded_relation_graph, plot_graph_and_page, compose_graphs
# import logging
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")

# General
# =======
flags.define_string('model_dir', '', 'Checkpoint containing the exported model')
flags.define_string('eval_list', '', '.lst-file specifying the dataset used for evaluation')
flags.define_integer('batch_size', 1, 'number of elements to be evaluated in each batch (default: %(default)s).')

# Model parameter
# ===============
flags.define_integer('num_classes', 2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_integer('sample_num_relations_to_consider', 100,
                     'number of sampled relations to be tested (half pos, half neg)')
flags.define_boolean('sample_relations', False, 'sample relations to consider or use full graph')

# Visual input
# ===============
flags.define_boolean('image_input', False,
                     'use image as additional input for GNN (visual input is calculated from image and regions)')
flags.define_boolean('assign_visual_features_to_nodes', True, 'use visual node features, only if image_input is True')
flags.define_boolean('assign_visual_features_to_edges', False, 'use visual edge features, only if image_input is True')
flags.define_boolean('mvn', True, 'MVN on the image input')

# Input function
# ===============
flags.define_dict('input_params', {}, "dict of key=value pairs defining the input configuration")

# Confidences & Clustering
# ========================
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration")
flags.define_boolean('mask_horizontally_separated_confs', False,
                     'set confidences of edges over horizontal separators, '
                     'whose nodes are also vertically and horizontally aligned, to zero.')
flags.define_boolean('mask_heading_separated_confs', False,
                     'set confidences of edges from regions (upper) to headings (lower), '
                     'whose nodes are also vertically and horizontally aligned, to zero.')
flags.define_float('mask_threshold', 0.3, 'only consider edges to mask with a confidence greater than this value')
flags.define_string("out_dir", "", "directory to save graph confidences jsons and clustering pageXMLs. It retains the "
                                   "folder structure of the input data. Use an empty 'out_dir' for the original folder")
flags.define_choices('save_conf', ['no_conf', 'with_conf', 'only_conf'], 'no_conf', str,
                     "('no_conf', 'with_conf', 'only_conf')", 'handles the saving of the graph confidences.')

# Misc
# ====
flags.define_integer('num_p_r_thresholds', 20, 'number of thresholds used for precision-recall-curve')
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_string("debug_dir", "", "directory to save debug outputs")
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")
flags.FLAGS.parse_flags()
flags.define_boolean("try_gpu", True if flags.FLAGS.gpu_devices != [] else False,
                     "try to load '<model>_gpu.pb' if possible")
flags.FLAGS.parse_flags()


class EvaluateRelation(object):
    def __init__(self):
        self._flags = flags.FLAGS
        self._params = {'num_gpus': len(self._flags.gpu_devices)}
        self._page_paths = None
        self._json_paths = None
        self._dataset = None
        self._dataset_iterator = None
        self._next_batch = None
        self._pb_path = None
        if self._flags.try_gpu:
            try:
                self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*_gpu.pb", "cpu"))
            except IOError:
                logger.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file")
        if not self._pb_path:
            self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*best*.pb", "_gpu.pb"))
        logger.info(f"pb_path is {self._pb_path}")
        self._input_fn = InputGNN(self._flags)
        self._tb_clustering = TextblockClustering(self._flags)
        # Print params
        flags.print_flags()
        self._input_fn.print_params()
        self._tb_clustering.print_params()

    def _load_graph(self):
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        with tf.io.gfile.GFile(self._pb_path, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we can use again a convenient built-in function to import a graph_def into the
        # current default Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, input_map=None, return_elements=None, name="",
                                producer_op_list=None)
        return graph

    def _get_placeholder(self):
        input_fn_params = self._input_fn.input_params

        ph_dict = dict()
        ph_dict['num_nodes'] = ph(tf.int32, [None], name='num_nodes')  # [batch_size]
        ph_dict['num_interacting_nodes'] = ph(tf.int32, [None], name='num_interacting_nodes')  # [batch_size]
        ph_dict['interacting_nodes'] = ph(tf.int32, [None, None, 2],
                                          name='interacting_nodes')  # [batch_size, max_num_interacting_nodes, 2]

        # add node features if present
        if 'node_feature_dim' in input_fn_params and input_fn_params["node_feature_dim"] > 0:
            # feature dim by masking
            if 'node_input_feature_mask' in input_fn_params:
                node_feature_dim = input_fn_params["node_input_feature_mask"].count(True) if \
                    input_fn_params["node_input_feature_mask"] else input_fn_params["node_feature_dim"]
            else:
                node_feature_dim = input_fn_params["node_feature_dim"]
            # [batch_size, max_num_nodes, node_feature_dim]
            ph_dict['node_features'] = ph(tf.float32, [None, None, node_feature_dim], name='node_features')

        # add edge features if present
        if 'edge_feature_dim' in input_fn_params and input_fn_params["edge_feature_dim"] > 0:
            # feature dim by masking
            if 'edge_input_feature_mask' in input_fn_params:
                edge_feature_dim = input_fn_params["edge_input_feature_mask"].count(True) if \
                    input_fn_params["edge_input_feature_mask"] else input_fn_params["edge_feature_dim"]
            else:
                edge_feature_dim = input_fn_params["edge_feature_dim"]
            # [batch_size, max_num_interacting_nodes, edge_feature_dim]
            ph_dict['edge_features'] = ph(tf.float32, [None, None, edge_feature_dim], name='edge_features')

        # add visual features
        if self._flags.image_input:
            if self._flags.assign_visual_features_to_nodes or self._flags.assign_visual_features_to_edges:
                img_channels = 1
                if 'load_mode' in input_fn_params and input_fn_params['load_mode'] == 'RGB':
                    img_channels = 3
                # [batch_size, pad_height, pad_width, channels] float
                ph_dict['image'] = ph(tf.float32, [None, None, None, img_channels], name="image")
                # [batch_size, 3] int
                ph_dict['image_shape'] = ph(tf.int32, [None, 3], name="image_shape")
                if self._flags.assign_visual_features_to_nodes:
                    # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes] float
                    ph_dict['visual_regions_nodes'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_nodes")
                    # [batch_size, max_num_nodes] int
                    ph_dict['num_points_visual_regions_nodes'] = ph(tf.int32, [None, None],
                                                                    name="num_points_visual_regions_nodes")
                if self._flags.assign_visual_features_to_edges:
                    # [batch_size, max_num_nodes, 2, max_num_points_visual_regions_edges] float
                    ph_dict['visual_regions_edges'] = ph(tf.float32, [None, None, 2, None], name="visual_regions_edges")
                    # [batch_size, max_num_nodes] int
                    ph_dict['num_points_visual_regions_edges'] = ph(tf.int32, [None, None],
                                                                    name="num_points_visual_regions_edges")
            else:
                logger.warning(f"Image_input was set to 'True', but no visual features were assigned. Specify flags.")

        # relations for evaluation
        # [batch_size, max_num_relations, num_relation_components]
        ph_dict['relations_to_consider_belong_to_same_instance'] = \
            ph(tf.int32, [None, None, self._flags.num_relation_components],
               name='relations_to_consider_belong_to_same_instance')
        # # [batch_size]
        # ph_dict['num_relations_to_consider_belong_to_same_instance'] = \
        #     ph(tf.int32, [None], name='num_relations_to_consider_belong_to_same_instance')
        return ph_dict

    def _mask_horizontally_separated_confs(self, confs, page_path):
        from citlab_article_separation.util.page_stats import get_text_region_article_dict
        timer = time.time()
        # get page information
        page = Page(page_path)
        text_region_articles = get_text_region_article_dict(page=page)
        num_masked_correct = 0
        num_masked_error = 0
        regions = page.get_regions()
        if self._flags.mask_horizontally_separated_confs and 'SeparatorRegion' not in regions and \
                not self._flags.mask_heading_separated_confs:
            logger.warning(f"No separators found for confidence masking.")
            return confs, np.zeros_like(confs)
        text_regions = regions['TextRegion']
        try:
            separator_regions = regions['SeparatorRegion']
        except KeyError:
            logger.warning(f"No separators found for confidence masking.")
            separator_regions = None
        if not len(confs) == len(text_regions):
            logger.info(f"Number of nodes in confidences ({len(confs)}) does not match number of "
                        f"text regions ({len(text_regions)}) in {page_path}.\nDiscarding text regions.")
            # discard text regions
            text_regions, _ = discard_text_regions_and_lines(text_regions)
        num_text_regions = len(text_regions)
        # compute mask
        mask = np.ones_like(confs, dtype=np.int32)
        mask_correct = np.zeros_like(confs, dtype=np.int32)
        mask_error = np.zeros_like(confs, dtype=np.int32)
        for i in range(num_text_regions):
            for j in range(i + 1, num_text_regions):

                if i == 24 and j == 30:
                    num_masked_correct += 1
                    mask_correct[i, j] = 1
                    mask_correct[j, i] = 1
                    mask[i, j] = 0
                    mask[j, i] = 0
                    continue

                # only mask confidences over certain threshold
                if confs[i, j] < self._flags.mask_threshold:
                    continue
                tr_i = text_regions[i]
                tr_j = text_regions[j]
                # mask edges over text regions which are vertically AND horizontally aligned (i.e. same column)
                # and where the lower text region is a heading
                if self._flags.mask_heading_separated_confs:
                    if is_aligned_heading_separated(tr_i, tr_j):
                        if text_region_articles[tr_i.id] == text_region_articles[tr_j.id]:
                            num_masked_error += 1
                            status = "FALSELY"
                            mask_error[i, j] = 1
                            mask_error[j, i] = 1
                        else:
                            num_masked_correct += 1
                            status = "CORRECTLY"
                            mask_correct[i, j] = 1
                            mask_correct[j, i] = 1
                        logger.debug(f"Pair ({i}, {j}) heading separated {status}. "
                                     f"Previous confs: ({i}, {j}) = {confs[i, j]:.4f}, ({j}, {i}) = {confs[j, i]:.4f}")
                        mask[i, j] = 0
                        mask[j, i] = 0
                        continue
                # mask edges over text regions which are vertically AND horizontally aligned (i.e. same column)
                # and separated by a horizontal separator region
                if self._flags.mask_horizontally_separated_confs and separator_regions:
                    if is_aligned_horizontally_separated(tr_i, tr_j, separator_regions):
                        if text_region_articles[tr_i.id] == text_region_articles[tr_j.id]:
                            num_masked_error += 1
                            status = "FALSELY"
                            mask_error[i, j] = 1
                            mask_error[j, i] = 1
                        else:
                            num_masked_correct += 1
                            status = "CORRECTLY"
                            mask_correct[i, j] = 1
                            mask_correct[j, i] = 1
                        logger.debug(f"Pair ({i}, {j}) horizontally separated {status}. "
                                     f"Previous confs: ({i}, {j}) = {confs[i, j]:.4f}, ({j}, {i}) = {confs[j, i]:.4f}")
                        mask[i, j] = 0
                        mask[j, i] = 0
        logger.debug(f"Masked {num_masked_error + num_masked_correct} confidences "
                     f"(correct = {num_masked_correct}, error = {num_masked_error}).")
        logger.info(f"Time for masking separated confidences = {time.time() - timer}.")
        masked = (-mask + 1) * confs  # the masked confidences
        masked_error = mask_error * confs
        masked_correct = mask_correct * confs
        return mask * confs, masked, masked_error, masked_correct

    def evaluate(self):
        logger.info("Start evaluation...")
        # Load graph
        graph = self._load_graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self._flags.gpu_devices)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.compat.v1.Session(graph=graph, config=session_config)

        with sess.graph.as_default() as graph:
            # for i in [n.name for n in tf.get_default_graph().as_graph_def().node if "graph" not in n.name]:
            #     logger.debug(i)
            with tf.Graph().as_default():  # write dummy placeholder in another graph
                placeholders = self._get_placeholder()
                # Placeholders:
                # 'num_nodes'             [batch_size]
                # 'num_interacting_nodes' [batch_size]
                # 'interacting_nodes'     [batch_size, max_num_interacting_nodes, 2]
                # 'node_features'         [batch_size, max_num_nodes, node_feature_dim]
                # 'edge_features'         [batch_size, max_num_interacting_nodes, edge_feature_dim]
                # 'image'                 [batch_size, pad_height, pad_width, channels]
                # 'image_shape'           [batch_size, 3]
                # 'visual_regions_nodes'  [batch_size, max_num_nodes, 2, max_num_points_visual_regions_nodes]
                # 'num_points_visual_regions_nodes' [batch_size, max_num_nodes]
                # 'visual_regions_edges'  [batch_size, max_num_nodes, 2, max_num_points_visual_regions_edges]
                # 'num_points_visual_regions_edges' [batch_size, max_num_nodes]
                # 'relations_to_consider_belong_to_same_instance' [batch_size, max_num_relations, num_relation_components]

            # Build dataset iterator
            self._dataset = self._input_fn.get_dataset()
            self._dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self._next_batch = self._dataset_iterator.get_next()
            with open(self._flags.eval_list, 'r') as eval_list_file:
                self._json_paths = [line.rstrip() for line in eval_list_file.readlines()]
                self._page_paths = [get_page_from_json_path(json_path) for json_path in self._json_paths]

            output_node = graph.get_tensor_by_name("output_belong_to_same_instance:0")
            target_key = "relations_to_consider_gt"
            targets = []  # gather targets for precision/recall evaluation
            probs = []  # gather probabilities for precision/recall evaluation
            feed_dict = {}

            batch_counter = 0
            start_timer = time.time()
            while True:  # Loop until dataset is empty
                if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch_counter:
                    logger.info(f"stop validation after {batch_counter} batches with "
                                f"{self._flags.batch_size} samples each.")
                    break
                try:
                    page_path = self._page_paths.pop(0)
                    json_path = self._json_paths.pop(0)

                    logger.info(f"Processing... {page_path}")
                    # Skip files where json is missing (e.g. when there are less than 2 text regions)
                    if not os.path.isfile(json_path):
                        logger.warning(f"No json file found to given pageXML {page_path}. Skipping.")
                        continue

                    # get one batch (input_dict, target_dict) from generator
                    next_batch = sess.run([self._next_batch])[0]
                    batch_counter += 1
                    target = next_batch[1][target_key]
                    targets.append(target)
                    # num_relations_to_consider = next_batch[0]["num_relations_to_consider_belong_to_same_instance"]

                    # print(os.path.basename(page_path), os.path.basename(json_path))
                    # if "19080115_1-0005" not in page_path:
                    #     continue

                    # assign placeholder_dict to feed_dict
                    for key in placeholders:
                        if type(placeholders[key]) == dict:
                            for i in placeholders[key]:
                                input_name = graph.get_tensor_by_name(placeholders[key][i].name)
                                feed_dict[input_name] = next_batch[0][key][i]
                        else:
                            input_name = graph.get_tensor_by_name(placeholders[key].name)
                            feed_dict[input_name] = next_batch[0][key]

                    # run model with feed_dict
                    output = sess.run(output_node, feed_dict=feed_dict)

                    # evaluate the output
                    class_probabilities = output[0, :, 1]
                    probs.append(class_probabilities)

                    if 'node_features' in placeholders:
                        node_features_node = graph.get_tensor_by_name('node_features:0')
                        node_features = feed_dict[node_features_node][0]  # assume batch_size = 1
                    if 'edge_features' in placeholders:
                        edge_features_node = graph.get_tensor_by_name('edge_features:0')
                        edge_features = feed_dict[edge_features_node][0]  # assume batch_size = 1

                    # clustering of confidence graph
                    confidences = np.reshape(class_probabilities, [node_features.shape[0], -1])
                    additional_info = ""
                    # Manually set confidences of edges over horizontal separators and headings to zero
                    if self._flags.mask_heading_separated_confs or self._flags.mask_horizontally_separated_confs:
                        if self._flags.mask_heading_separated_confs:
                            additional_info += "_maskHead"
                        if self._flags.mask_horizontally_separated_confs:
                            additional_info += "_maskSep"
                        confidences, masked_confs, masked_confs_error, masked_confs_correct = \
                            self._mask_horizontally_separated_confs(confidences, page_path)

                    if self._flags.save_conf != 'no_conf':
                        # save confidences
                        save_conf_to_json(confidences=confidences,
                                          page_path=page_path,
                                          save_dir=self._flags.out_dir)
                        if self._flags.save_conf == 'only_conf':
                            # skip clustering
                            continue
                    self._tb_clustering.set_debug(page_path, self._flags.debug_dir)
                    self._tb_clustering.set_confs(confidences)
                    self._tb_clustering.calc(method=self._flags.clustering_method)
                    # exit()
                    # continue

                    # save pageXMLs with new clusterings
                    cluster_path = save_clustering_to_page(clustering=self._tb_clustering.tb_labels,
                                                           page_path=page_path,
                                                           save_dir=self._flags.out_dir,
                                                           info=self._tb_clustering.get_info(
                                                               self._flags.clustering_method) + additional_info)
                    # info = self._tb_clustering.get_info(self._flags.clustering_method)
                    # save_name = re.sub(r'\.xml$', '_clustering.xml', os.path.basename(os.path.relpath(page_path)))
                    # page_dir = re.sub(r'page$', 'clustering', os.path.dirname(os.path.relpath(page_path)))
                    # save_dir = self._flags.out_dir
                    # if info:
                    #     save_dir = os.path.join(save_dir, page_dir, info)
                    # else:
                    #     save_dir = os.path.join(save_dir, page_dir)
                    # cluster_path = os.path.join(save_dir, save_name)

                    # debug output
                    # TODO: add more debug images for (corrects/falses/targets/predictions etc.)
                    if self._flags.debug_dir:
                        if not os.path.isdir(self._flags.debug_dir):
                            os.makedirs(self._flags.debug_dir)

                        relations_node = graph.get_tensor_by_name('relations_to_consider_belong_to_same_instance:0')
                        relations = feed_dict[relations_node][0]  # assume batch_size = 1

                        # if 'edge_features' in placeholders:
                        #     feature_dicts = [{'separated': bool(e)} for e in edge_features[:, :1].flatten()]
                        #     graph_full = build_weighted_relation_graph(relations.tolist(),
                        #                                                class_probabilities.tolist(),
                        #                                                feature_dicts)
                        # else:
                        nx_graph = build_thresholded_relation_graph(relations, confidences,
                                                                    self._tb_clustering.clustering_params[
                                                                        "confidence_threshold"])

                        # mask_graph = build_thresholded_relation_graph(relations, masked_confs,
                        #                                               self._flags.mask_threshold)
                        #
                        # mask_error_graph = build_thresholded_relation_graph(relations, masked_confs_error,
                        #                                                     self._flags.mask_threshold)
                        # for u, v, d in mask_error_graph.edges(data=True):
                        #     d["weight"] *= -1
                        #
                        # mask_correct_graph = build_thresholded_relation_graph(relations, masked_confs_correct,
                        #                                                       self._flags.mask_threshold)
                        #
                        # mask_compose_graph = compose_graphs([mask_error_graph, mask_correct_graph])

                        # # full confidence graph
                        # edge_colors = []
                        # for u, v, d in graph_full.edges(data='weight'):
                        #     edge_colors.append(d)
                        # plot_graph_and_page(page_path=page_path,
                        #                     graph=graph_full,
                        #                     node_features=node_features,
                        #                     save_dir=self._flags.debug_dir,
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     desc='confidences',
                        #                     edge_color=edge_colors,
                        #                     edge_cmap=plt.get_cmap('jet'),
                        #                     edge_vmin=0.0,
                        #                     edge_vmax=1.0)
                        # # confidence histogram
                        # plot_confidence_histogram(class_probabilities, 10, page_path, self._flags.debug_dir,
                        #                           desc='conf_hist')

                        # clustered graph
                        edge_colors = []
                        for u, v, d in nx_graph.edges(data='weight'):
                            edge_colors.append(d)
                        plot_graph_clustering_and_page(graph=nx_graph,
                                                       node_features=node_features,
                                                       page_path=page_path,
                                                       cluster_path=cluster_path,
                                                       save_dir=self._flags.debug_dir,
                                                       threshold=self._tb_clustering.clustering_params[
                                                           "confidence_threshold"],
                                                       info=self._tb_clustering.get_info(
                                                           self._flags.clustering_method) + additional_info,
                                                       with_edges=True,
                                                       with_labels=True,
                                                       edge_color=edge_colors,
                                                       edge_cmap=plt.get_cmap('jet'))

                        # # masked graph
                        # edge_colors = []
                        # for u, v, d in mask_graph.edges(data='weight'):
                        #     edge_colors.append(d)
                        # plot_graph_and_page(graph=mask_graph,
                        #                     node_features=node_features,
                        #                     page_path=page_path,
                        #                     save_dir=self._flags.debug_dir,
                        #                     threshold=self._flags.mask_threshold,
                        #                     info=self._tb_clustering.get_info(
                        #                         self._flags.clustering_method) + additional_info,
                        #                     name="conf_mask",
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     edge_color=edge_colors,
                        #                     edge_cmap=plt.get_cmap('jet'))
                        #
                        # # masked error graph
                        # plot_graph_and_page(graph=mask_error_graph,
                        #                     node_features=node_features,
                        #                     page_path=page_path,
                        #                     save_dir=self._flags.debug_dir,
                        #                     threshold=self._flags.mask_threshold,
                        #                     info=self._tb_clustering.get_info(
                        #                         self._flags.clustering_method) + additional_info,
                        #                     name="conf_mask_error",
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     edge_color="r")
                        # # masked correct graph
                        # plot_graph_and_page(graph=mask_correct_graph,
                        #                     node_features=node_features,
                        #                     page_path=page_path,
                        #                     save_dir=self._flags.debug_dir,
                        #                     threshold=self._flags.mask_threshold,
                        #                     info=self._tb_clustering.get_info(
                        #                         self._flags.clustering_method) + additional_info,
                        #                     name="conf_mask_correct",
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     edge_color="g")
                        # # masked compose graph
                        # edge_colors = []
                        # for u, v, d in mask_compose_graph.edges(data='weight'):
                        #     if d > 0:
                        #         edge_colors.append("g")
                        #     else:
                        #         edge_colors.append("r")
                        # plot_graph_and_page(graph=mask_compose_graph,
                        #                     node_features=node_features,
                        #                     page_path=page_path,
                        #                     save_dir=self._flags.debug_dir,
                        #                     threshold=self._flags.mask_threshold,
                        #                     info=self._tb_clustering.get_info(
                        #                         self._flags.clustering_method) + additional_info,
                        #                     name="conf_mask_compose",
                        #                     with_edges=True,
                        #                     with_labels=True,
                        #                     edge_color=edge_colors)

                        # final graph
                        edge_colors = []
                        for u, v, d in nx_graph.edges(data='weight'):
                            edge_colors.append(d)
                        plot_graph_and_page(graph=nx_graph,
                                            node_features=node_features,
                                            page_path=page_path,
                                            save_dir=self._flags.debug_dir,
                                            threshold=self._tb_clustering.clustering_params[
                                                           "confidence_threshold"],
                                            info=self._tb_clustering.get_info(
                                                self._flags.clustering_method) + additional_info,
                                            name="masked_graph",
                                            with_edges=True,
                                            with_labels=True,
                                            edge_color=edge_colors,
                                            edge_cmap=plt.get_cmap('jet'))

                # break as soon as dataset is empty
                # (IndexError for empty page_paths list, OutOfRangeError for empty tf dataset)
                except (tf.errors.OutOfRangeError, IndexError) as ex:
                    # logger.error(traceback.format_exc())
                    break

            # # Compute Precision, Recall, F1
            # full_targets = np.squeeze(np.concatenate(targets, axis=-1))
            # full_probs = np.squeeze(np.concatenate(probs, axis=-1))
            # prec, rec, thresholds = precision_recall_curve(full_targets, full_probs)
            # f_score = (2 * prec * rec) / (prec + rec)  # element-wise (broadcast)
            #
            # # P, R, F at relative thresholds
            # print("\n Relative Thresholds:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # for j in range(self._flags.num_p_r_thresholds + 1):
            #     i = j * ((len(thresholds) - 1) // self._flags.num_p_r_thresholds)
            #     print(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")
            #
            # # P, R, F at fixed thresholds
            # print("\n Fixed Thresholds:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # step = 1 / self._flags.num_p_r_thresholds
            # j = 0
            # for i in range(len(thresholds)):
            #     if thresholds[i] >= j * step:
            #         print(f" |{thresholds[i]:10f}{prec[i]:12f}{rec[i]:12f}{f_score[i]:12f}")
            #         j += 1
            #         if j * step >= 1.0:
            #             break
            #
            # # Best F1-Score
            # i_f = np.argmax(f_score)
            # print("\n Best F1-Score:")
            # print(f" |{'Threshold':>10}{'Precision':>12}{'Recall':>12}{'F1-Score':>12}")
            # print(" | " + "-" * 45)
            # print(f" |{thresholds[i_f]:10f}{prec[i_f]:12f}{rec[i_f]:12f}{f_score[i_f]:12f}")

            logger.info("Time: {:.2f} seconds".format(time.time() - start_timer))
        logger.info("Evaluation finished.")


if __name__ == "__main__":
    logger.setLevel("DEBUG")
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    eval_rel = EvaluateRelation()
    eval_rel.evaluate()
