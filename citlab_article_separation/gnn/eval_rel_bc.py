import os
import logging
import time
import re
import functools
import numpy as np
from scipy.stats import gmean
import util.flags as flags
import networkx as nx
import json
import tensorflow as tf
from copy import deepcopy
from util.misc import get_path_from_exportdir
from util.graph_util import load_graph
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from input_fn.input_fn_rel.external.python.textblock_clustering import TextblockClustering

# Model imports
import model_fn.model_fn_rel.model_fn_relation as models
from model_fn.model_fn_rel.util_model_fn_rel.rel_clustering import DBScanRelation
from input_fn.input_fn_rel.input_fn_generator_relation import InputFnRelation, get_img_from_page_path, get_page_from_json_path
from citlab_python_util.parser.xml.page.page import Page
import citlab_python_util.parser.xml.page.plot as plot_util

# General
# =======
flags.define_string('model_dir', '', 'Checkpoint containing the exported model')
flags.define_string('model_type', 'ModelRelation', "Type of model (currently only 'ModelRelation')")
flags.define_string('val_list', '', '.lst-file specifying the dataset used for validation')
flags.define_integer('val_batch_size', 1, 'number of elements to be evaluated in each batch (default: %(default)s).')

# Model parameter
# ===============
flags.define_list('classifiers', str, 'STR', 'list of classifier names', ['belong_to_same_instance'])
flags.define_list('num_classes_per_classifier', int, 'INT',
                  'list of number of classes (including garbage class) for each classifier', [2])
flags.define_list('num_relation_components_per_classifier', int, 'INT',
                  'list of number of components of the associated relations for each classifier', [2])
flags.define_boolean('sample_relations', False, 'sample relations to consider or use full graph')
flags.define_list('sample_num_relations_to_consider_per_classifier', int, 'INT',
                  'list of number of sampled relations to be tested (half pos, half neg) for each classifier', [100])

# Visual features
# ===============
flags.define_boolean('image_input', False, 'use image as additional input for GNN (visual features are '
                                           'calculated from image and regions)')
flags.define_string('backbone', 'ARU_v1', 'Backbone graph to use.')
flags.define_boolean('mvn', True, 'MVN on the input.')
flags.define_dict('graph_backbone_params', {}, "key=value pairs defining the configuration of the backbone."
                                               "Backbone parametrization")
# E.g. train script: --feature_map_generation_params from_layer=[Mixed_5d,,Mixed_6e,Mixed_7c] layer_depth=[-1,128,-1,-1]
flags.define_dict('feature_map_generation_params',
                  {'layer_depth': [-1, -1, -1]},
                  "key=value pairs defining the configuration of the feature map generation."
                  "FeatureMap Generator parametrization, see main graph, e.g. model_fn.model_fn_objdet.graphs.main.SSD")

# Input function
# ===============
flags.define_boolean('create_data_from_pagexml', False, 'generate input data on the fly from the pagexml or load'
                                                        'it directly from json')
flags.define_choices('interaction_from_pagexml', ['fully', 'delaunay'], 'fully', str, "('fully', 'delaunay')",
                     'determines the setup of the interacting_nodes when loading from pagexml.')
flags.define_dict('input_params', {}, "dict of key=value pairs defining the configuration of the input.")

# Clustering
# ==========
flags.define_string('clustering_config', './input_fn/input_fn_rel/external/python/tb_clustering.cfg',
                    'path to config file containing clustering parameters')
flags.define_choices('clustering_method', ['greedy', 'dbscan', 'linkage', 'dbscanStd'], 'dbscan', str,
                     "('greedy', 'dbscan', 'linkage', 'dbscanStd')",
                     'cluster algorithm to be used on graph output for articles')
flags.define_float('edge_weight_threshold', 0.5, 'edge weight threshold for graph creation')
flags.define_float('cluster_agreement', 0.5, 'cluster agreement threshold for graph clustering')
flags.define_choices('weight_handling', ['avg', 'min', 'max'], 'avg', str, "('avg', 'min', 'max')",
                     "handles how the confidences of an edge pair are combined in the undirected graph")

# Misc
# ====
flags.define_list('gpu_devices', int, 'INT', 'list of GPU indices to use. ', [])
flags.define_float('gpu_memory_fraction', 0.95, 'set between 0.1 and 1, value - 0.09 is passed to session_config, to '
                                                'take overhead in account, smaller val_batch_size may needed, '
                                                '(default: %(default)s)')
flags.define_string("cluster_dir", "cluster_output", "directory to save graph clustering pageXMLs")
flags.define_string("debug_dir", "debug_output", "directory to save debug outputs")
flags.define_integer("batch_limiter", -1, "set to positiv value to stop validation after this number of batches")
flags.FLAGS.parse_flags()
flags.define_boolean("try_gpu", True if flags.FLAGS.gpu_devices != [] else False,
                     "try to load '<model>_gpu.pb' if possible")
flags.FLAGS.parse_flags()


class EvaluateRelation(object):
    def __init__(self):
        self._flags = flags.FLAGS
        flags.print_flags()
        self._params = {'num_gpus': len(self._flags.gpu_devices)}
        self._page_paths = None
        self._val_dataset = None
        self._val_dataset_iterator = None
        self._next_batch = None
        self._pb_path = None
        if self._flags.try_gpu:
            try:
                self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*_gpu.pb"))
            except IOError:
                logging.warning("Could not find gpu-model-pb-file, continue with cpu-model-pb-file")
        if not self._pb_path:
            self._pb_path = os.path.join(get_path_from_exportdir(self._flags.model_dir, "*.pb", "_gpu.pb"))
        print("pb_path", self._pb_path)
        self._input_fn_generator = InputFnRelation(self._flags)
        self._model = getattr(models, self._flags.model_type)(self._params)

    def evaluate(self):
        print("Start evaluation...")
        graph = load_graph(self._pb_path)
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in self._flags.gpu_devices)
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=self._flags.gpu_memory_fraction - 0.09,
                                              allow_growth=False)  # - 0.09 for memory overhead
        session_config = tf.compat.v1.ConfigProto(
            gpu_options=gpu_options,
            allow_soft_placement=True)
        sess = tf.compat.v1.Session(graph=graph, config=session_config)

        with sess.graph.as_default() as graph:
            # for i in [n.name for n in tf.get_default_graph().as_graph_def().node]:
            #     print(i)

            with tf.Graph().as_default():  # write dummy placeholder in another graph
                placeholders = self._model.get_placeholder()
            # Placeholders:
            # 'num_nodes'             [batch_size]
            # 'num_interacting_nodes' [batch_size]
            # 'interacting_nodes'     [batch_size, max_num_interacting_nodes, 2]
            # 'node_features'         [batch_size, max_num_nodes, node_feature_dim]
            # 'edge_features'         [batch_size, max_num_interacting_nodes, edge_feature_dim]
            # 'relations_to_consider' = dict()
            # for each classifier_name:
            #   'relations_to_consider[classifier_name]' = 'relations_to_consider_' + classifier_name
            #   [batch_size, max_num_relations, num_relation_components]

            self._val_dataset = self._input_fn_generator.get_input_fn_val()
            self._val_dataset_iterator = tf.compat.v1.data.make_one_shot_iterator(self._val_dataset)
            self._next_batch = self._val_dataset_iterator.get_next()
            with open(self._flags.val_list, 'r') as val_list_file:
                self._page_paths = [line.rstrip() for line in val_list_file.readlines()]
                if not self._flags.create_data_from_pagexml:
                    self._page_paths = [get_page_from_json_path(json_path) for json_path in self._page_paths]

            output_nodes_names = self._model.get_output_nodes(has_graph=False).split(",")
            output_nodes = [graph.get_tensor_by_name(x + ":0") for x in output_nodes_names]
            output_nodes_dict = {}

            target_keys = self._model.get_target_keys().split(",")
            target_dict = {}
            feed_dict = {}

            sum_true_pos = dict()
            sum_false_pos = dict()
            sum_true_neg = dict()
            sum_false_neg = dict()
            for classifier in self._flags.classifiers:
                sum_true_pos[classifier] = 0.0
                sum_false_pos[classifier] = 0.0
                sum_true_neg[classifier] = 0.0
                sum_false_neg[classifier] = 0.0

            confidence_graphs = dict()
            for classifier_name in self._flags.classifiers:
                confidence_graphs[classifier_name] = dict()

            # initialize textblock clustering
            tb_clustering = TextblockClustering(self._flags.clustering_config)

            batch_counter = 0
            start_timer = time.time()
            while True:  # Loop until val_dataset is empty
                if self._flags.batch_limiter != -1 and self._flags.batch_limiter <= batch_counter:
                    print("stop validation after {} batches with {} samples each.".format(batch_counter,
                                                                                          self._flags.val_batch_size))
                    break
                try:
                    # get one batch (input_dict, target_dict) from generator
                    next_batch = sess.run([self._next_batch])[0]
                    batch_counter += 1
                    page_path = self._page_paths.pop(0)

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
                    output_nodes_res = sess.run(output_nodes, feed_dict=feed_dict)
                    # contains a list of output tensors in order as in the return string of model.get_output_nodes

                    # assign targets and model outputs to dicts for the models print_evaluate function
                    for key in target_keys:
                        target_dict[key] = next_batch[1][key]
                    for key, value in zip(output_nodes_names, output_nodes_res):
                        output_nodes_dict[key] = value

                    # evaluate the output
                    print(f"\nProcessing... {page_path}")
                    if 'node_features' in placeholders:
                        node_features_node = graph.get_tensor_by_name('node_features:0')
                        node_features = feed_dict[node_features_node][0]  # assume batch_size = 1
                    if 'edge_features' in placeholders:
                        edge_features_node = graph.get_tensor_by_name('edge_features:0')
                        edge_features = feed_dict[edge_features_node][0]  # assume batch_size = 1

                    for classifier_name in self._flags.classifiers:
                        # print(f"Classifier: {classifier_name}")
                        relations_node = graph.get_tensor_by_name('relations_to_consider_' + classifier_name + ':0')
                        relations = feed_dict[relations_node][0]  # assume batch_size = 1
                        output = output_nodes_dict['output_' + classifier_name]
                        # target = target_dict['relations_to_consider_gt'][classifier_name]
                        # prediction = np.argmax(output, axis=-1)  # [batch_size, max_num_relations]
                        class_probabilities = output[0, :, 1]

                        # # update confidence graphs dict with current output
                        # confidence_graphs[classifier_name].update(
                        #     build_confidence_graph_dict(graph=graph_full, page_path=page_path))
                        # continue

                        # clustering of confidence graph
                        confidences = np.reshape(class_probabilities, [node_features.shape[0], -1])
                        tb_clustering.set_confs(confidences, symmetry_fn=gmean)
                        tb_clustering.calc(method=self._flags.clustering_method)

                        # save pageXMLs with new clusterings
                        cluster_dir = os.path.join(self._flags.cluster_dir, classifier_name)
                        cluster_path = save_clustering_to_page(clustering=tb_clustering.tb_labels,
                                                               page_path=page_path,
                                                               save_dir=cluster_dir,
                                                               threshold=self._flags.edge_weight_threshold,
                                                               agreement=self._flags.cluster_agreement,
                                                               weight_handling=self._flags.weight_handling)

                        # debug output
                        # TODO: add more debug images for (corrects/falses/targets/predictions etc.)
                        if self._flags.debug_dir:
                            if not os.path.isdir(self._flags.debug_dir):
                                os.makedirs(self._flags.debug_dir)

                            # if 'edge_features' in placeholders:
                            #     feature_dicts = [{'separated': bool(e)} for e in edge_features[:, :1].flatten()]
                            #     graph_full = build_weighted_relation_graph(relations.tolist(),
                            #                                                class_probabilities.tolist(),
                            #                                                feature_dicts)
                            # else:
                            graph_full = build_weighted_relation_graph(relations.tolist(),
                                                                       class_probabilities.tolist())

                            graph_u = create_undirected_graph(graph_full,
                                                              weight_handling=self._flags.weight_handling,
                                                              reciprocal=False)
                            edges_below_threshold = [(u, v) for u, v, w in graph_u.edges.data('weight')
                                                     if w < self._flags.edge_weight_threshold]
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
                                                           threshold=self._flags.edge_weight_threshold,
                                                           agreement=self._flags.cluster_agreement,
                                                           weight_handling=self._flags.weight_handling,
                                                           with_edges=True,
                                                           with_labels=True,
                                                           edge_color=edge_colors,
                                                           edge_cmap=plt.get_cmap('jet'))

                    eval_dict = self._model.print_evaluate(output_nodes_dict, target_dict,
                                                           next_batch[0]["num_relations_to_consider"])
                    for classifier in self._flags.classifiers:
                        sum_true_pos[classifier] += eval_dict[classifier]['true_pos'].tolist()
                        sum_false_pos[classifier] += eval_dict[classifier]['false_pos'].tolist()
                        sum_true_neg[classifier] += eval_dict[classifier]['true_neg'].tolist()
                        sum_false_neg[classifier] += eval_dict[classifier]['false_neg'].tolist()

                    if batch_counter > 0:
                        break

                # break as soon as val_dataset is empty
                except tf.errors.OutOfRangeError:
                    break

            # # save confidence graphs as json
            # save_name = os.path.splitext(os.path.basename(self._flags.val_list))[0]
            # save_name = re.sub(r'json[0-9]+[df][0-9]+v?', 'confidences', save_name)
            # for classifier_name in self._flags.classifiers:
            #     save_dir = os.path.join(self._flags.cluster_dir, classifier_name)
            #     if not os.path.isdir(save_dir):
            #         os.makedirs(save_dir)
            #     save_path = os.path.join(save_dir, save_name + ".json")
            #     with open(save_path, "w") as out_file:
            #         # TODO: save space by reducing floating points?
            #         json.dump(confidence_graphs[classifier_name], out_file)#, indent=4)
            #         print(f"Wrote json dump: {save_path}.")

            # compute binary classification metrics (for each classifier)
            print("SUMMARY:")
            for classifier in self._flags.classifiers:
                # TODO: Check for divisions by zero
                tp = sum_true_pos[classifier]
                fp = sum_false_pos[classifier]
                tn = sum_true_neg[classifier]
                fn = sum_false_neg[classifier]
                p = tp + fp
                n = tn + fn
                accuracy = (tp + tn) / (p + n)
                precision = tp / p
                recall = tp / (tp + fn)
                fscore = (2 * precision * recall) / (precision + recall)
                print("Classifier: '{}'".format(classifier))
                print("  {:9s} = {:.3f}".format("Accuracy", accuracy))
                print("  {:9s} = {:.3f}".format("Precision", precision))
                print("  {:9s} = {:.3f}".format("Recall", recall))
                print("  {:9s} = {:.3f}".format("F1-Score", fscore))

            print("Time: {:.2f} seconds".format(time.time() - start_timer))
            self._model.print_evaluate_summary()
        print("Evaluation finished.")


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


def create_undirected_graph(digraph, weight_handling='avg', reciprocal=False):
    assert weight_handling in ('avg', 'min', 'max'), \
        f"Weight handling {weight_handling} is not supported! Choose from ('avg', 'min', 'max')"

    G = nx.Graph()
    G.graph.update(deepcopy(digraph.graph))
    G.add_nodes_from((n, deepcopy(d)) for n, d in digraph._node.items())
    for u, successcors in digraph.succ.items():
        for v, data in successcors.items():
            u_v_data = deepcopy(data)
            if v in digraph.pred[u]:  # reciprocal (both edges present)
                # edge data handling
                v_u_data = digraph.pred[u][v]
                if weight_handling == 'avg':
                    u_v_data['weight'] = (u_v_data['weight'] + v_u_data['weight']) / 2
                elif weight_handling == 'max':
                    u_v_data['weight'] = max(u_v_data['weight'], v_u_data['weight'])
                elif weight_handling == 'min':
                    u_v_data['weight'] = min(u_v_data['weight'], v_u_data['weight'])
                G.add_edge(u, v, **u_v_data)
            elif not reciprocal:
                G.add_edge(u, v, **u_v_data)
    return G


def discard_regions(text_regions):
    discard = 0
    for tr in text_regions:
        # ... without text lines
        if not tr.text_lines:
            text_regions.remove(tr)
            logging.debug(f"Discarding TextRegion {tr.id} (no textlines)")
            discard += 1
        # ... too small
        bounding_box = tr.points.to_polygon().get_bounding_box()
        if bounding_box.width < 10 or bounding_box.height < 10:
            text_regions.remove(tr)
            logging.debug(f"Discarding TextRegion {tr.id} (bounding box too small, height={bounding_box.height}, width={bounding_box.width})")
            logging.debug(f"TextRegion {tr.id} has {len(tr.text_lines)} assigned TextLine(s)")
            for tl in tr.text_lines:
                logging.debug(f" TextLine {tl.id}:")
                logging.debug(f"  Baseline: height = {tl.baseline.to_polygon().get_bounding_box().height}, "
                              f"width = {tl.baseline.to_polygon().get_bounding_box().width}")
                logging.debug(f"  SurrPoly: height = {tl.surr_p.to_polygon().get_bounding_box().height}, "
                              f"width = {tl.surr_p.to_polygon().get_bounding_box().width}")
                logging.debug(f"  Text: \"{tl.text}\"")
            discard += 1
    if discard > 0:
        logging.warning(f"Discarded {discard} degenerate text_region(s). Either no text lines or region too small.")
    return text_regions


def build_confidence_graph_dict(graph, page_path):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions = discard_regions(text_regions)
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


def save_clustering_to_page(clustering, page_path, save_dir, threshold=None, agreement=None, weight_handling=None):
    page = Page(page_path)
    text_regions = page.get_regions()['TextRegion']
    text_regions = discard_regions(text_regions)
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
    t = '' if threshold is None else f't{threshold}'
    a = '' if agreement is None else f'a{agreement}'
    h = '' if weight_handling is None else f'{weight_handling}'
    save_name = re.sub(r'\.xml$', '_clustering.xml', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{t}_{a}_{h}")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, save_name)
    page.write_page_xml(save_path)
    logging.info(f"Saved pageXML with graph clustering '{save_path}'")
    return save_path


def graph_edge_conf_histogram(graph, num_bins):
    conf = [w for u, v, w in graph.edges.data('weight')]
    c = np.array(conf)
    print(c.shape)
    print(np.min(c))
    print(np.max(c))
    print(np.mean(c))
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


def plot_graph_clustering_and_page(graph, node_features, page_path, cluster_path, save_dir, threshold=None,
                                   agreement=None, weight_handling=None, with_edges=True, with_labels=True, **kwds):
    # Get pagexml and image file
    original_page = Page(page_path)
    img_path = get_img_from_page_path(page_path)
    cluster_page = Page(cluster_path)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 9))
    fig.canvas.set_window_title(img_path)
    axes[0].set_title('ground truth')
    axes[1].set_title(f'graph_t{threshold}_a{agreement}_{weight_handling}')
    axes[2].set_title(f'clustering_t{threshold}_a{agreement}_{weight_handling}')

    # Plot GT page and Cluster page
    plot_util.plot_pagexml(original_page, img_path, ax=axes[0], plot_article=True, plot_legend=False)
    plot_util.plot_pagexml(cluster_page, img_path, ax=axes[2], plot_article=True, plot_legend=False)
    for ax in axes:
        ax.tick_params(
            axis='both',
            which='both',
            bottom=False,
            left=False,
            labelbottom=False,
            labelleft=False)

    # Add graph to subplot
    # Get node positions from features
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
    node_collection = nx.draw_networkx_nodes(graph, positions, ax=axes[1], node_color=node_colors, node_size=50, **kwds)
    node_collection.set_zorder(3)
    graph_views['nodes'] = [node_collection]
    # Draw edges
    if with_edges:
        edge_collection = nx.draw_networkx_edges(graph, positions, ax=axes[1], width=0.5, arrows=False, **kwds)
        if edge_collection is not None:
            edge_collection.set_zorder(2)
            graph_views['edges'] = [edge_collection]
        # optional colorbar
        if 'edge_cmap' in kwds and 'edge_color' in kwds:
            norm = Normalize(vmin=min(kwds['edge_color']), vmax=max(kwds['edge_color']))
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), ax=axes[1])
    # Draw labels
    if with_labels:
        label_collection = nx.draw_networkx_labels(graph, positions, ax=axes[1], font_size=5, **kwds)
        graph_views['labels'] = [label_collection]
    plt.connect('key_press_event', lambda event: toggle_graph_view(event, graph_views))
    # Draw page underneath
    plot_util.plot_pagexml(original_page, img_path, ax=axes[1], plot_article=True, plot_legend=False)

    # Save image
    t = '' if threshold is None else f't{threshold}'
    a = '' if agreement is None else f'a{agreement}'
    h = '' if weight_handling is None else f'{weight_handling}'
    save_name = re.sub(r'\.xml$', f'_clustering_debug.jpg', os.path.basename(page_path))
    page_dir = os.path.dirname(page_path)
    save_dir = os.path.join(save_dir, page_dir, f"{t}_{a}_{h}")
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
    # Get node positions from features
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
            fig.colorbar(ScalarMappable(norm=norm, cmap=kwds['edge_cmap']), ax=ax)
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
