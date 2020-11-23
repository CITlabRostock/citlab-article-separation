import logging
import time
import functools
import os
import json
import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.compat.v1 import placeholder as ph
from sklearn.metrics import precision_recall_curve
from scipy.stats import gmean
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable

import citlab_python_util.basic.flags as flags
from citlab_python_util.io.path_util import *
from citlab_python_util.parser.xml.page.page import Page
import citlab_python_util.parser.xml.page.plot as plot_util

from citlab_article_separation.gnn.input.input_dataset import InputGNN
from citlab_article_separation.gnn.clustering.textblock_clustering import TextblockClustering
from citlab_article_separation.gnn.input.feature_generation import discard_text_regions_and_lines as discard_regions


# General
# =======
flags.define_string('model_dir',  '', 'Checkpoint containing the exported model')
flags.define_string('eval_list',   '', '.lst-file specifying the dataset used for evaluation')
flags.define_integer('batch_size', 1, 'number of elements to be evaluated in each batch (default: %(default)s).')

# Model parameter
# ===============
flags.define_integer('num_classes',             2, 'number of classes (including garbage class)')
flags.define_integer('num_relation_components', 2, 'number of components of the associated relations')
flags.define_boolean('sample_relations',    False, 'sample relations to consider or use full graph')
flags.define_integer('sample_num_relations_to_consider', 100,
                     'number of sampled relations to be tested (half pos, half neg)')

# Visual input
# ===============
flags.define_boolean('image_input', False,
                     'use image as additional input for GNN (visual input is calculated from image and regions)')
flags.define_boolean('assign_visual_features_to_nodes', True, 'use visual node features, only if image_input is True')
flags.define_boolean('assign_visual_features_to_edges', False, 'use visual edge features, only if image_input is True')
flags.define_boolean('mvn', True, 'MVN on the image input')

# Input function
# ===============
flags.define_boolean('create_data_from_pagexml', False,
                     'generate input data on the fly from the pagexml or load it directly from json')
flags.define_choices('interaction_from_pagexml', ['fully', 'delaunay'], 'fully', str, "('fully', 'delaunay')",
                     'determines the setup of the interacting_nodes when loading from pagexml.')
flags.define_dict('input_params', {}, "dict of key=value pairs defining the input configuration")

# Confidences & Clustering
# ========================
flags.define_choices('clustering_method', ['dbscan', 'linkage', 'greedy', 'dbscan_std'], 'dbscan', str,
                     "('dbscan', 'linkage', 'greedy', 'dbscan_std')", 'clustering method to be used')
flags.define_dict('clustering_params', {}, "dict of key=value pairs defining the clustering configuration")
flags.define_string("out_dir", "", "directory to save graph confidences jsons and clustering pageXMLs. It retains the "
                                   "folder structure of the input data. Use an empty 'out_dir' for the original folder")
flags.define_boolean("only_save_conf", False, "Only save the graph confidences and skip the clustering process")

# Misc
# ====
flags.define_integer('num_p_r_thresholds', 20, 'number of thresholds used for precision-recall-curve')
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_string("debug_dir", "debug_output", "directory to save debug outputs")
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
        self._dataset = None
        self._dataset_iterator = None
        self._next_batch = None
        self._pb_path = None
        if self._flags.try_gpu:
            try:
                self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*_gpu.pb", "cpu"))
            except IOError:
                logging.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file")
        if not self._pb_path:
            self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*best*.pb", "_gpu.pb"))
        logging.info(f"pb_path is {self._pb_path}")
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
        ph_dict['interacting_nodes'] = ph(tf.int32, [None, None, 2], name='interacting_nodes')  # [batch_size, max_num_interacting_nodes, 2]

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
                logging.warning(f"Image_input was set to 'True', but no visual features were assigned. Specify flags.")

        # relations for evaluation
        # [batch_size, max_num_relations, num_relation_components]
        ph_dict['relations_to_consider_belong_to_same_instance'] = \
            ph(tf.int32, [None, None, self._flags.num_relation_components],
               name='relations_to_consider_belong_to_same_instance')
        # # [batch_size]
        # ph_dict['num_relations_to_consider_belong_to_same_instance'] = \
        #     ph(tf.int32, [None], name='num_relations_to_consider_belong_to_same_instance')
        return ph_dict

    def evaluate(self):
        logging.info("Start evaluation...")
        graph = self._load_graph()
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self._flags.gpu_devices)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
        sess = tf.compat.v1.Session(graph=graph, config=session_config)

        with sess.graph.as_default() as graph:
            # for i in [n.name for n in tf.get_default_graph().as_graph_def().node if "graph" not in n.name]:
            #     logging.debug(i)

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

            self._dataset = self._input_fn.get_dataset()
            self._dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self._dataset)
            self._next_batch = self._dataset_iterator.get_next()
            with open(self._flags.eval_list, 'r') as eval_list_file:
                self._page_paths = [line.rstrip() for line in eval_list_file.readlines()]
                if not self._flags.create_data_from_pagexml:
                    self._page_paths = [get_page_from_json_path(json_path) for json_path in self._page_paths]

            output_node = graph.get_tensor_by_name("output_belong_to_same_instance:0")
            target_key = "relations_to_consider_gt"
            targets = []  # gather targets for precision/recall evaluation
            probs = []  # gather probabilities for precision/recall evaluation
            feed_dict = {}

            batch_counter = 0
            start_timer = time.time()
            while True:  # Loop until dataset is empty
                if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch_counter:
                    logging.info(f"stop validation after {batch_counter} batches with "
                                 f"{self._flags.batch_size} samples each.")
                    break
                try:
                    # get one batch (input_dict, target_dict) from generator
                    next_batch = sess.run([self._next_batch])[0]
                    batch_counter += 1
                    page_path = self._page_paths.pop(0)
                    target = next_batch[1][target_key]
                    targets.append(target)
                    # num_relations_to_consider = next_batch[0]["num_relations_to_consider_belong_to_same_instance"]

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

                    logging.info(f"Processing... {page_path}")
                    if 'node_features' in placeholders:
                        node_features_node = graph.get_tensor_by_name('node_features:0')
                        node_features = feed_dict[node_features_node][0]  # assume batch_size = 1
                    if 'edge_features' in placeholders:
                        edge_features_node = graph.get_tensor_by_name('edge_features:0')
                        edge_features = feed_dict[edge_features_node][0]  # assume batch_size = 1

                    # clustering of confidence graph
                    confidences = np.reshape(class_probabilities, [node_features.shape[0], -1])

                    if self._flags.only_save_conf:
                        # save confidences
                        save_conf_to_json(confidences=confidences,
                                          page_path=page_path,
                                          save_dir=self._flags.out_dir)
                        # skip clustering
                        continue

                    self._tb_clustering.set_confs(confidences)
                    self._tb_clustering.calc(method=self._flags.clustering_method)

                    # save pageXMLs with new clusterings
                    cluster_path = save_clustering_to_page(clustering=self._tb_clustering.tb_labels,
                                                           page_path=page_path,
                                                           save_dir=self._flags.out_dir,
                                                           info=self._tb_clustering.get_info(self._flags.clustering_method))

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
                        graph_full = build_weighted_relation_graph(relations.tolist(),
                                                                   class_probabilities.tolist())

                        graph_u = create_undirected_graph(graph_full,
                                                          reciprocal=False)
                        edges_below_threshold = [(u, v) for u, v, w in graph_u.edges.data('weight')
                                                 if w < self._tb_clustering.clustering_params["confidence_threshold"]]
                        graph_u.remove_edges_from(edges_below_threshold)

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
                        for u, v, d in graph_u.edges(data='weight'):
                            edge_colors.append(d)
                        plot_graph_clustering_and_page(graph=graph_u,
                                                       node_features=node_features,
                                                       page_path=page_path,
                                                       cluster_path=cluster_path,
                                                       save_dir=self._flags.debug_dir,
                                                       threshold=self._tb_clustering.clustering_params["confidence_threshold"],
                                                       info=self._tb_clustering.get_info(self._flags.clustering_method),
                                                       with_edges=True,
                                                       with_labels=True,
                                                       edge_color=edge_colors,
                                                       edge_cmap=plt.get_cmap('jet'))

                # break as soon as dataset is empty
                except tf.errors.OutOfRangeError:
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

            logging.info("Time: {:.2f} seconds".format(time.time() - start_timer))
        logging.info("Evaluation finished.")


def build_weighted_relation_graph(relations, weights, feature_dicts=None):
    if type(relations) == np.ndarray:
        assert relations.ndim == 2, f"Expected 'relations' to be 2d, got {relations.ndim}d."
        relations = relations.tolist()
    if type(weights) == np.ndarray:
        assert weights.ndim == 1, f"Expected 'weights' to be 1d, got {weights.ndim}d."
        weights = weights.tolist()
    assert(len(relations) == len(weights)), f"Number of elements in 'relations' {len(relations)} and" \
                                            f" 'weights' {len(weights)} has to match."

    graph = nx.DiGraph()
    for i in range(len(relations)):
        if feature_dicts is not None:
            graph.add_edge(*relations[i], weight=weights[i], **feature_dicts[i])
        else:
            graph.add_edge(*relations[i], weight=weights[i])
    return graph


def create_undirected_graph(digraph, symmetry_fn=gmean, reciprocal=False):
    G = nx.Graph()
    G.graph.update(deepcopy(digraph.graph))
    G.add_nodes_from((n, deepcopy(d)) for n, d in digraph._node.items())
    for u, successcors in digraph.succ.items():
        for v, data in successcors.items():
            u_v_data = deepcopy(data)
            if v in digraph.pred[u]:  # reciprocal (both edges present)
                # edge data handling
                v_u_data = digraph.pred[u][v]
                if symmetry_fn:
                    u_v_data['weight'] = symmetry_fn([u_v_data['weight'], v_u_data['weight']])
                G.add_edge(u, v, **u_v_data)
            elif not reciprocal:
                G.add_edge(u, v, **u_v_data)
    return G


def build_confidence_graph_dict(graph, page_path):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions, _ = discard_regions(text_regions)
    assert graph.number_of_nodes() == len(text_regions), \
        f"Number of nodes in graph ({graph.number_of_nodes()}) does not match number of text regions " \
        f"({len(text_regions)}) in {page_path}."

    out_dict = dict()
    page_name = os.path.basename(page_path)
    out_dict[page_name] = dict()

    for text_region in text_regions:
        out_dict[page_name][text_region.id] = {'article_id': None, 'confidences': dict()}

    for i, j, w in graph.edges.data('weight'):
        out_dict[page_name][text_regions[i].id]['confidences'][text_regions[j].id] = w

    return out_dict


def save_conf_to_json(confidences, page_path, save_dir, symmetry_fn=gmean):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions, _ = discard_regions(text_regions)
    assert len(confidences) == len(text_regions), f"Number of nodes in confidences ({len(confidences)}) does not " \
                                                  f"match number of text regions ({len(text_regions)}) in {page_path}."

    # make confidences symmetric
    if symmetry_fn:
        conf_transpose = confidences.transpose()
        temp_mat = np.stack([confidences, conf_transpose], axis=-1)
        confidences = symmetry_fn(temp_mat, axis=-1)

    # Build confidence dict
    conf_dict = dict()
    for i in range(len(text_regions)):
        conf_dict[text_regions[i].id] = dict()
        for j in range(len(text_regions)):
            conf_dict[text_regions[i].id][text_regions[j].id] = str(confidences[i, j])
    out_dict = dict()
    out_dict["confidences"] = conf_dict

    # Dump json
    save_name = os.path.splitext(os.path.basename(page_path))[0] + "_confidences.json"
    page_dir = re.sub(r'page$', 'confidences', os.path.dirname(page_path))
    save_dir = os.path.join(save_dir, page_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    with open(save_path, "w") as out_file:
        json.dump(out_dict, out_file)
        logging.info(f"Saved json with graph confidences '{save_path}'")


def save_clustering_to_page(clustering, page_path, save_dir, info=""):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions, _ = discard_regions(text_regions)
    assert len(clustering) == len(text_regions), f"Number of nodes in clustering ({len(clustering)}) does not " \
                                                 f"match number of text regions ({len(text_regions)}) in {page_path}."

    # Set textline article ids based on clustering
    textlines = []
    for index, text_region in enumerate(text_regions):
        article_id = clustering[index]
        for text_line in text_region.text_lines:
            text_line.set_article_id("a" + str(article_id))
            textlines.append(text_line)

    # Write pagexml
    page.set_textline_attr(textlines)
    save_name = re.sub(r'\.xml$', '_clustering.xml', os.path.basename(page_path))
    page_dir = re.sub(r'page$', 'clustering', os.path.dirname(page_path))
    if info:
        save_dir = os.path.join(save_dir, page_dir, info)
    else:
        save_dir = os.path.join(save_dir, page_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    page.write_page_xml(save_path)
    logging.info(f"Saved pageXML with graph clustering '{save_path}'")
    return save_path


def graph_edge_conf_histogram(graph, num_bins):
    conf = [w for u, v, w in graph.edges.data('weight')]
    c = np.array(conf)
    hist, bin_edges = np.histogram(conf, bins=num_bins, range=(0.0, 1.0))
    for i in range(num_bins):
        logging.debug(f"Edges with conf [{bin_edges[i]:.2f}, {bin_edges[i + 1]:.2f}): {hist[i]}")
    plt.hist(conf, num_bins, range=(0.0, 1.0), rwidth=0.98)
    plt.xticks(np.arange(0, 1.01, 0.1))
    plt.show()


def plot_confidence_histogram(confidences, bins, page_path, save_dir, desc=None):
    # Get img path
    img_path = get_img_from_page_path(page_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    ax.set_xticks(np.arange(0, 1.01, 0.1))

    # Plot histogram
    counts, bins, _ = ax.hist(confidences, bins=bins, range=(0.0, 1.0), rwidth=0.98)

    # Save image
    desc = '' if desc is None else f'_{desc}'
    save_name = re.sub(r'\.xml$', f'{desc}.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{desc}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


def plot_graph_clustering_and_page(graph, node_features, page_path, cluster_path, save_dir,
                                   threshold, info, with_edges=True, with_labels=True, **kwds):
    # Get pagexml and image file
    original_page = Page(page_path)
    img_path = get_img_from_page_path(page_path)
    cluster_page = Page(cluster_path)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    axes[0].set_title(f'GT_with_graph_conf_threshold{threshold}')
    axes[1].set_title(f'cluster_{info}')

    # Plot Cluster page
    plot_util.plot_pagexml(cluster_page, img_path, ax=axes[1], plot_article=True, plot_legend=False)
    for ax in axes:
        ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add graph to subplot
    # Get node positions from input
    page_resolution = original_page.get_image_resolution()
    region_centers = node_features[:, 2:4] * page_resolution
    positions = dict()
    for n in range(node_features.shape[0]):
        positions[n] = region_centers[n]
        # node_positions[n][1] = page_resolution[1] - node_positions[n][1]  # reverse y-values (for plotting)

    # Get article colors according to baselines
    article_dict = original_page.get_article_dict()
    unique_ids = sorted(set(article_dict.keys()), key=functools.cmp_to_key(compare_article_ids))
    if None in unique_ids:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids) - 1] + [plot_util.DEFAULT_COLOR]))
    else:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids)]))
    # Build node colors (region article_id matching baselines article_id)
    region_article_ids = get_region_article_ids(original_page)
    node_colors = [article_colors[a_id] for a_id in region_article_ids]
    node_colors = [node_colors[i] for i in list(graph.nodes)]  # reorder coloring according to node_list

    # Draw nodes
    graph_views = dict()
    node_collection = nx.draw_networkx_nodes(graph, positions, ax=axes[0], node_color=node_colors, node_size=50, **kwds)
    node_collection.set_zorder(3)
    graph_views['nodes'] = [node_collection]
    # Draw edges
    if with_edges:
        edge_collection = nx.draw_networkx_edges(graph, positions, ax=axes[0], width=0.5, arrows=False, **kwds)
        if edge_collection is not None:
            edge_collection.set_zorder(2)
            graph_views['edges'] = [edge_collection]
        # optional colorbar
        if 'edge_cmap' in kwds and 'edge_color' in kwds:
            # add colorbar to confidence graph
            divider = make_axes_locatable(axes[0])
            cax = divider.append_axes("right", size="5%", pad=0.05)
            # norm = Normalize(vmin=min(kwds['edge_color']), vmax=max(kwds['edge_color']))
            norm = Normalize(vmin=threshold, vmax=1.0)
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), cax=cax, format="%.2f")
            # hacky way to get same size adjustment on other two images
            divider = make_axes_locatable(axes[1])
            cax = divider.append_axes("left", size="5%", pad=0.05)
            cax.axis("off")
    # Draw labels
    if with_labels:
        label_collection = nx.draw_networkx_labels(graph, positions, ax=axes[0], font_size=5, **kwds)
        graph_views['labels'] = [label_collection]
    plt.connect('key_press_event', lambda event: toggle_graph_view(event, graph_views))
    # Draw page underneath
    plot_util.plot_pagexml(original_page, img_path, ax=axes[0], plot_article=True, plot_legend=False)

    # Save image
    save_name = re.sub(r'\.xml$', f'_clustering_debug.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    if info:
        save_dir = os.path.join(save_dir, page_dir, info)
    else:
        save_dir = os.path.join(save_dir, page_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


def plot_graph_and_page(page_path, graph, node_features, save_dir,
                        with_edges=True, with_labels=True, desc=None, **kwds):
    # Get pagexml and image file
    page = Page(page_path)
    img_path = get_img_from_page_path(page_path)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    ax.tick_params(axis='both', which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    # Add graph to subplot
    # Get node positions from input
    page_resolution = page.get_image_resolution()
    region_centers = node_features[:, 2:4] * page_resolution
    positions = dict()
    for n in range(node_features.shape[0]):
        positions[n] = region_centers[n]
        # node_positions[n][1] = page_resolution[1] - node_positions[n][1]  # reverse y-values (for plotting)

    # Get article colors according to baselines
    article_dict = page.get_article_dict()
    unique_ids = sorted(set(article_dict.keys()), key=functools.cmp_to_key(compare_article_ids))
    if None in unique_ids:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids) - 1] + [plot_util.DEFAULT_COLOR]))
    else:
        article_colors = dict(zip(unique_ids, plot_util.COLORS[:len(unique_ids)]))
    # Build node colors (region article_id matching baselines article_id)
    region_article_ids = get_region_article_ids(page)
    node_colors = [article_colors[a_id] for a_id in region_article_ids]
    node_colors = [node_colors[i] for i in list(graph.nodes)]  # reorder coloring according to node_list

    # Draw nodes
    graph_views = dict()
    node_collection = nx.draw_networkx_nodes(graph, positions, ax=ax, node_color=node_colors, node_size=50, **kwds)
    node_collection.set_zorder(3)
    graph_views['nodes'] = [node_collection]
    # Draw edges
    if with_edges:
        edge_collection = nx.draw_networkx_edges(graph, positions, ax=ax, width=0.5, arrows=False, **kwds)
        if edge_collection is not None:
            edge_collection.set_zorder(2)
            graph_views['edges'] = [edge_collection]
        # optional colorbar
        if 'edge_cmap' in kwds and 'edge_vmin' in kwds and 'edge_vmax' in kwds:
            norm = Normalize(vmin=kwds['edge_vmin'], vmax=kwds['edge_vmax'])
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), ax=ax, fraction=0.046, pad=0.04)
    # Draw labels
    if with_labels:
        label_collection = nx.draw_networkx_labels(graph, positions, ax=ax, font_size=5, **kwds)
        graph_views['labels'] = [label_collection]
    # Draw page underneath
        plot_util.plot_pagexml(page, img_path, ax=ax, plot_article=True, plot_legend=False)

    plt.connect('key_press_event', lambda event: toggle_graph_view(event, graph_views))

    # Save image
    desc = '' if desc is None else f'_{desc}'
    save_name = re.sub(r'\.xml$', f'{desc}.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{desc}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=1000)
    logging.info(f"Saved debug image '{save_path}'")
    plt.close(plt.gcf())


def compare_article_ids(a, b):
    # assume article IDs are named like "a<INTEGER>"
    if a is None and b is None:
        return 0
    elif a is None:
        return 1
    elif b is None:
        return -1
    elif int(a[1:]) < int(b[1:]):
        return -1
    elif int(a[1:]) == int(b[1:]):
        return 0
    else:
        return 1


def toggle_graph_view(event, views):
    """Switch between different views in the current plot by pressing the ``event`` key.

    :param event: the key event given by the user, various options available, e.g. to toggle the edges
    :param views: dictionary of different views given by name:object pairs
    :type event: matplotlib.backend_bases.KeyEvent
    :return: None
    """
    # Toggle nodes with optional labels
    if event.key == 'c' and "nodes" in views:
        for node_collection in views["nodes"]:
            is_visible = node_collection.get_visible()
            node_collection.set_visible(not is_visible)
        if "labels" in views:
            for label_collection in views["labels"]:
                for label in label_collection.values():
                    is_visible = label.get_visible()
                    label.set_visible(not is_visible)
        plt.draw()
    # Toggle edges
    if event.key == 'e' and "edges" in views:
        for edge_collection in views["edges"]:
            is_visible = edge_collection.get_visible()
            edge_collection.set_visible(not is_visible)
        plt.draw()
    if event.key == 'h':
        print("\tc: toggle graph nodes\n"
              "\te: toggle graph edges\n")
    else:
        return


def get_region_article_ids(page):
    assert type(page) == Page, f"Expected object of type 'Page', got {type(page)}."
    text_regions = page.get_regions()['TextRegion']
    text_regions_article_id = []
    for text_region in text_regions:
        # get all article_ids for textlines in this region
        tr_article_ids = []
        for text_line in text_region.text_lines:
            tr_article_ids.append(text_line.get_article_id())
        # count article_id occurences
        unique_article_ids = list(set(tr_article_ids))
        article_id_occurences = np.array([tr_article_ids.count(a_id) for a_id in unique_article_ids], dtype=np.int32)
        # assign article_id by majority vote
        if article_id_occurences.shape[0] > 1:
            assign_index = np.argmax(article_id_occurences)
            assign_article_id = unique_article_ids[int(assign_index)]
            text_regions_article_id.append(assign_article_id)
        else:
            text_regions_article_id.append(unique_article_ids[0])
    return text_regions_article_id


if __name__ == "__main__":
    logging.getLogger().setLevel('INFO')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    eval_rel = EvaluateRelation()
    eval_rel.evaluate()
