import os
import argparse
import json
import re
import logging
import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError
from shapely.geometry import LineString

from citlab_python_util.math.rounding import round_by_precision_and_base as round_base
from citlab_python_util.io.path_util import get_img_from_page_path
from citlab_article_separation.gnn.input.textblock_similarity import TextblockSimilarity
from citlab_python_util.image_processing.swt_dist_trafo import StrokeWidthDistanceTransform
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.geometry.util import convex_hull, bounding_box


def get_text_region_geometric_features(text_region, norm_x, norm_y):
    tr_points = np.array(text_region.points.points_list, dtype=np.int32)
    # bounding box of text region
    min_x = np.min(tr_points[:, 0])
    min_y = np.min(tr_points[:, 1])
    max_x = np.max(tr_points[:, 0])
    max_y = np.max(tr_points[:, 1])
    size_x = float(max_x) - float(min_x)
    size_y = float(max_y) - float(min_y)
    # feature vector describing the extension of the text region
    region_size_x = size_x / norm_x
    region_size_y = size_y / norm_y
    # feature vector describing the center of the text region
    region_center_x = (min_x + size_x / 2) / norm_x
    region_center_y = (min_y + size_y / 2) / norm_y
    # 4-dimensional feature
    return [region_size_x, region_size_y, region_center_x, region_center_y]


def get_text_region_baseline_features(text_region, norm_x, norm_y):
    feature = []
    # geometric information about top & bottom textline of text region
    top_baseline = text_region.text_lines[0].baseline.to_polygon()
    bottom_baseline = text_region.text_lines[-1].baseline.to_polygon()
    for poly in (top_baseline, bottom_baseline):
        # bounding box of baseline
        min_x = min(poly.x_points)
        max_x = max(poly.x_points)
        min_y = min(poly.y_points)
        max_y = max(poly.y_points)
        size_x = float(max_x) - float(min_x)
        size_y = float(max_y) - float(min_y)
        # feature vector describing the extension of the baseline
        bl_size_x = size_x / norm_x
        bl_size_y = size_y / norm_y
        # feature vector describing the center of the baseline
        bl_center_x = sum(poly.x_points) / (norm_x * len(poly.x_points))
        bl_center_y = sum(poly.y_points) / (norm_y * len(poly.y_points))
        # extend feature
        feature.extend([bl_size_x, bl_size_y, bl_center_x, bl_center_y])
    # return 8-dimensional feature
    return feature


def get_text_region_punctuation_feature(text_region):
    # 1-feature for empty text regions
    if all([not line.text for line in text_region.text_lines]):
        return [1.0, 1.0]
    # context information about beginning and end of text region (character based)
    # shift top textline until it contains text
    index_top = 0
    while not text_region.text_lines[index_top].text:
        index_top += 1
    top_textline = text_region.text_lines[index_top]
    # does it start with a capital letter
    starts_upper = float(top_textline.text[0].isupper())
    # shift bottom textline until it contains text
    index_bottom = -1
    while not text_region.text_lines[index_bottom].text:
        index_bottom -= 1
    bottom_textline = text_region.text_lines[index_bottom]
    # does it end on an eos punctuation mark
    ends_eos_punctuation = float(bottom_textline.text[-1] in ('.', '!', '?'))
    # return 2-dimensional feature
    return [starts_upper, ends_eos_punctuation]


def get_tb_similarities(text_regions, feature_extractor):
    # build {tb : text} dict
    tb_dict = dict()
    for text_region in text_regions:
        text = "\n".join([text_line.text for text_line in text_region.text_lines])
        tb_dict[text_region.id] = text
    # run feature extractor
    feature_extractor.set_tb_dict(tb_dict)
    feature_extractor.run()
    return feature_extractor.feature_dict


def get_textline_stroke_widths_heights_dist_trafo(page_path, text_lines):
    img_path = get_img_from_page_path(page_path)
    if not img_path:
        raise ValueError(f"Could not find corresponding image file to pagexml\n{page_path}")
    # initialize SWT and textline labeling
    SWT = StrokeWidthDistanceTransform(dark_on_bright=True)
    swt_img = SWT.distance_transform(img_path)
    textline_stroke_widths = dict()
    textline_heights = dict()
    for text_line in text_lines:
        # build surrounding polygons over text lines
        bounding_box = text_line.surr_p.to_polygon().get_bounding_box()
        xa, xb = bounding_box.x, bounding_box.x + bounding_box.width
        ya, yb = bounding_box.y, bounding_box.y + bounding_box.height
        # get swt for text line
        text_line_swt = swt_img[ya:yb + 1, xa:xb + 1]
        # get connected components in text line
        text_line_ccs = SWT.connected_components_cv(text_line_swt)
        text_line_ccs = SWT.clean_connected_components(text_line_ccs)
        # go over connected components to estimate stroke width and text height of the text line
        swt_cc_values = []
        text_line_height = 0
        for cc in text_line_ccs:
            # component is a 4-tuple (x, y, width, height)
            # take max value in distance_transform as stroke_width for current CC (can be 0)
            swt_cc_values.append(np.max(text_line_swt[cc[1]: cc[1] + cc[3], cc[0]: cc[0] + cc[2]]))
            # new text height
            if cc[3] > text_line_height:
                text_line_height = cc[3]

        textline_stroke_widths[text_line.id] = np.median(swt_cc_values) if swt_cc_values else 0.0
        textline_heights[text_line.id] = text_line_height
    return textline_stroke_widths, textline_heights


# def get_textline_stroke_widths_heights(page_path, text_lines):
#     # we need the corresponding image
#     img_endings = [".jpg", ".jpeg", ".png", ".tif"]
#     base_name = os.path.splitext(os.path.basename(page_path))[0]
#     img_name = os.path.join(os.path.dirname(page_path), "..", base_name)
#     img_path = None
#     if any([img_name.endswith(ending) for ending in img_endings]):
#         img_path = img_name
#     else:
#         for ending in img_endings:
#             if os.path.isfile(img_name + ending):
#                 img_path = img_name + ending
#                 break
#     if not img_path:
#         raise ValueError(f"Could not find corresponding image file to pagexml\n{page_path}")
#     # initialize SWT and textline labeling
#     SWT = StrokeWidthTransform(dark_on_bright=True, canny_auto='otsu')
#     swt, ccs = SWT.apply_fast_swt(img_path)
#     # t0 = time.time()
#     # build surrounding polygons over textlines
#     textline_stroke_widths = dict()
#     textline_heights = dict()
#     for text_line in text_lines:
#         bounding_box = text_line.surr_p.to_polygon().get_bounding_box()
#         xa, xb = bounding_box.x, bounding_box.x + bounding_box.width
#         ya, yb = bounding_box.y, bounding_box.y + bounding_box.height
#         text_line_swt = swt[ya:yb + 1, xa:xb + 1]
#         # compute mean swt over values > 0 (discard background)
#         if np.any(text_line_swt):
#             swt_value = np.mean(text_line_swt[text_line_swt > 0])
#         else:
#             # all swt values are 0, we can't compute the mean over an empty slice
#             swt_value = 0.0
#         textline_stroke_widths[text_line.id] = swt_value
#         # go over connected components to estimate text height of the text line
#         text_line_height = 0
#         for cc in ccs:
#             # component is a 4-tuple (x, y, width, height)
#             if bounding_box.contains_rectangle(Rectangle(int(cc[0]), int(cc[1]), int(cc[2]), int(cc[3]))):
#                 # new text height
#                 if cc[3] > text_line_height:
#                     text_line_height = cc[3]
#                 # each component can only occur once on the page
#                 ccs.remove(cc)
#         textline_heights[text_line.id] = text_line_height
#     # print(f"Time stats = {time.time() - t0}")
#     return textline_stroke_widths, textline_heights


def get_text_region_stroke_width_feature(text_region, textline_stroke_widths, norm=1.0):
    # 0-feature for empty text regions
    if all([not line.text for line in text_region.text_lines]):
        return [0.0]
    # maximum stroke width over text lines of text region
    # we prefer the maximum, so headings that are clustered in a block with other text dont get averaged out
    else:
        text_region_stroke_widths = [textline_stroke_widths[line.id] for line in text_region.text_lines if line.text]
        text_region_stroke_width = np.max(text_region_stroke_widths) / norm
        return [text_region_stroke_width]


def get_text_region_text_height_feature(text_region, textline_heights, norm=1.0):
    # 0-feature for empty text regions
    if all([not line.text for line in text_region.text_lines]):
        return [0.0]
    # maximum text height over text lines of text region
    # we prefer the maximum, so headings that are clustered in a block with other text dont get averaged out
    else:
        text_region_line_heights = [textline_heights[line.id] for line in text_region.text_lines if line.text]
        text_region_text_height = np.max(text_region_line_heights) / norm
        return [text_region_text_height]


def get_text_region_heading_feature(text_region):
    contains_heading = False
    if text_region.region_type.lower() == 'heading' or text_region.region_type.lower() == 'header':
        contains_heading = True
    # # check if any of the text regions text lines are marked as heading
    # for text_line in text_region.text_lines:
    #     if text_line.get_heading():
    #         contains_heading = True
    #         break
    return [float(contains_heading)]


def get_edge_separator_feature(text_region_a, text_region_b, seperator_regions):
    # surrounding polygons of text regions
    points_a = np.asarray(text_region_a.points.points_list, dtype=np.int32)
    points_b = np.asarray(text_region_b.points.points_list, dtype=np.int32)
    # bounding boxes of text regions
    min_x_a, max_x_a = np.min(points_a[:, 0]), np.max(points_a[:, 0])
    min_y_a, max_y_a = np.min(points_a[:, 1]), np.max(points_a[:, 1])
    width_a = float(max_x_a - min_x_a)
    height_a = float(max_y_a - min_y_a)
    min_x_b, max_x_b = np.min(points_b[:, 0]), np.max(points_b[:, 0])
    min_y_b, max_y_b = np.min(points_b[:, 1]), np.max(points_b[:, 1])
    width_b = float(max_x_b - min_x_b)
    height_b = float(max_y_b - min_y_b)
    # center of text regions
    center_x_a = min_x_a + width_a / 2
    center_y_a = min_y_a + height_a / 2
    center_x_b = min_x_b + width_b / 2
    center_y_b = min_y_b + height_b / 2
    # visual line connecting both text regions
    text_region_segment = LineString([(center_x_a, center_y_a), (center_x_b, center_y_b)])
    # go over seperator regions and check for intersections
    horizontally_separated = False
    vertically_separated = False
    for separator_region in seperator_regions:
        # surrounding polygon of separator region
        points_s = np.asarray(separator_region.points.points_list, dtype=np.int32)
        # bounding box of separator region
        min_x_s, max_x_s = np.min(points_s[:, 0]), np.max(points_s[:, 0])
        min_y_s, max_y_s = np.min(points_s[:, 1]), np.max(points_s[:, 1])
        # height-width-ratio of bounding box
        width = max(max_x_s - min_x_s, 1)
        height = max(max_y_s - min_y_s, 1)
        ratio = float(height) / float(width)
        # corner points of bounding box
        s1 = (min_x_s, min_y_s)
        s2 = (max_x_s, min_y_s)
        s3 = (min_x_s, max_y_s)
        s4 = (max_x_s, max_y_s)
        # check for intersections between text_region_line and bounding box as prior test
        if line_poly_intersection(text_region_segment, [s1, s2, s3, s4]):
            # check for intersections between text_region_line and surrounding polygon
            if line_poly_intersection(text_region_segment, separator_region.points.points_list):
                if ratio < 5:
                    horizontally_separated = True
                else:
                    vertically_separated = True
                # print(f"{text_region_a.id} - {text_region_b.id} separated by {separator_region.id}")
                if horizontally_separated and vertically_separated:
                    break
    return [float(horizontally_separated), float(vertically_separated)]


def line_poly_intersection(line, polygon):
    # Optionally close polygon
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    # Go over polygon segments and check for intersection with line
    for i in range(len(polygon) - 1):
        p1 = polygon[i]
        p2 = polygon[i + 1]
        segment = LineString([p1, p2])
        if line.intersects(segment):
            return True
    return False


def get_node_visual_regions(text_region):
    # surrounding polygon of text region
    points = text_region.points.points_list
    # bounding box
    bb = bounding_box(points)
    return bb


def get_edge_visual_regions(text_region_a, text_region_b):
    # TODO: Alternative to convex hull over regions?
    # surrounding polygons of text regions
    points_a = text_region_a.points.points_list
    points_b = text_region_b.points.points_list
    # convex hull over both regions
    hull = convex_hull(points_a + points_b)
    return hull


def get_page_feature_stats(page_path):
    # load page data
    regions, text_lines, polygons, article_ids, resolution = get_data_from_pagexml(page_path)
    text_regions = regions['TextRegion']

    # pre-compute stroke-width and height over textlines
    textline_stroke_widths, textline_heights = get_textline_stroke_widths_heights_dist_trafo(page_path, text_lines)

    # compute maximum stroke width and text height
    max_sw = 0
    max_th = 0
    for tr in text_regions:
        tr_sw = get_text_region_stroke_width_feature(tr, textline_stroke_widths, norm=1.0)[0]
        tr_th = get_text_region_text_height_feature(tr, textline_heights, norm=1.0)[0]
        if tr_sw > max_sw:
            max_sw = tr_sw
        if tr_th > max_th:
            max_th = tr_th
    return max_sw, max_th


def fully_connected_edges(num_nodes):
    node_indices = np.arange(num_nodes, dtype=np.int32)
    node_indices = np.tile(node_indices, [num_nodes, 1])
    node_indices_t = np.transpose(node_indices)
    # fully-connected
    interacting_nodes = np.stack([node_indices_t, node_indices], axis=2).reshape([-1, 2])
    # remove self-loops
    del_indices = np.arange(num_nodes) * (num_nodes + 1)
    interacting_nodes = np.delete(interacting_nodes, del_indices, axis=0)
    return interacting_nodes


def delaunay_edges(num_nodes, node_features, norm_x, norm_y):
    # region centers
    center_points = np.array(node_features, dtype=np.float32)[:, 2:4] * [norm_x, norm_y]
    # round to nearest 50px for a more homogenous layout
    center_points_smooth = round_base(center_points, base=50)
    # interacting nodes are neighbours in the delaunay triangulation
    try:
        delaunay = Delaunay(center_points_smooth)
    except QhullError:
        logging.warning("Delaunay input has the same x-coords. Defaulting to unsmoothed data.")
        delaunay = Delaunay(center_points)
    indice_pointer, indices = delaunay.vertex_neighbor_vertices
    interacting_nodes = []
    for v in range(num_nodes):
        neighbors = indices[indice_pointer[v]:indice_pointer[v + 1]]
        interaction = np.stack(np.broadcast_arrays(v, neighbors), axis=1)
        interacting_nodes.append(interaction)
    interacting_nodes = np.concatenate(interacting_nodes, axis=0)
    return interacting_nodes


def build_input_and_target_bc(page_path,
                              external_data=(),
                              interaction='delaunay',
                              visual_regions=False,
                              sim_feat_extractor=None):
    """ Computation of the input and target values to solve the AS problem with a graph neural network on BC level.

    :param page_path: path to page_xml
    :param external_data: list of additonal feature dicts from external json sources
    :param interaction: interacting_nodes setup
    :param visual_regions: optionally build visual regions for nodes and edges (used for visual feature extraction)
    :param sim_feat_extractor: feature extractor used for text block similarities
    :return: 'num_nodes', 'interacting_nodes', 'num_interacting_nodes' ,'node_features', 'edge_features',
     'gt_relations', 'gt_num_relations'
    """
    assert interaction in ('fully', 'delaunay'), \
        f"Interaction setup {interaction} is not supported. Choose from ('fully', 'delaunay') instead."

    # load page data
    regions, text_lines, polygons, article_ids, resolution = get_data_from_pagexml(page_path)
    text_regions = regions['TextRegion']

    # discard regions
    discard = 0
    text_lines_to_remove = []
    for tr in text_regions.copy():
        # ... without text lines
        if not tr.text_lines:
            text_regions.remove(tr)
            discard += 1
            continue
        # ... too small
        bounding_box = tr.points.to_polygon().get_bounding_box()
        if bounding_box.width < 5 or bounding_box.height < 5:
            text_regions.remove(tr)
            for text_line in tr.text_lines:
                text_lines_to_remove.append(text_line.id)
            discard += 1
    # discard corresponding text lines
    if text_lines_to_remove:
        text_lines = [line for line in text_lines if line.id not in text_lines_to_remove]
    if discard > 0:
        logging.warning(f"Discarded {discard} degenerate text_region(s). Either no text lines or region too small.")

    # number of nodes
    num_nodes = len(text_regions)
    if num_nodes <= 1:
        return None, None, None, None, None, None, None, None, None, None, None

    # node features
    node_features = []
    norm_x, norm_y = float(resolution[0]), float(resolution[1])

    # pre-compute stroke width and height over textlines (and their maximum value for normalization)
    textline_stroke_widths, textline_heights = get_textline_stroke_widths_heights_dist_trafo(page_path, text_lines)
    sw_max = np.max(list(textline_stroke_widths.values()))
    th_max = np.max(list(textline_heights.values()))

    # compute region features
    for text_region in text_regions:
        node_feature = []
        # region geometric feature (4-dim)
        node_feature.extend(get_text_region_geometric_features(text_region, norm_x, norm_y))
        # top/bottom baseline geometric feature (8-dim)
        node_feature.extend(get_text_region_baseline_features(text_region, norm_x, norm_y))
        # # punctuation feature (2-dim)
        # node_feature.extend(get_text_region_punctuation_feature(text_region))
        # stroke width feature (1-dim)
        node_feature.extend(get_text_region_stroke_width_feature(text_region, textline_stroke_widths, norm=sw_max))
        # text height feature (1-dim)
        node_feature.extend(get_text_region_text_height_feature(text_region, textline_heights, norm=th_max))
        # heading feature (1-dim)
        node_feature.extend(get_text_region_heading_feature(text_region))
        # external features
        for ext in external_data:
            try:
                ext_page = ext[os.path.basename(page_path)]
            except KeyError:
                logging.warning(f'Could not find key {page_path} in external data json.')
                continue
            if 'node_features' in ext_page:
                try:
                    node_feature.extend(ext['node_features'][text_region.id])
                except KeyError:
                    node_feature.extend([ext['node_features']['default']])
        # final node feature vector
        node_features.append(node_feature)

    # interacting nodes (currently fully-connected and delaunay triangulation)
    if interaction == 'fully' or num_nodes < 4:
        interacting_nodes = fully_connected_edges(num_nodes)
    elif interaction == 'delaunay':
        interacting_nodes = delaunay_edges(num_nodes, node_features, norm_x, norm_y)

    # number of interacting nodes
    num_interacting_nodes = interacting_nodes.shape[0]

    # edge features for each pair of interacting nodes
    edge_features = []

    # pre-compute text block similarities with word vectors
    tb_sim_dict = get_tb_similarities(text_regions, sim_feat_extractor) if sim_feat_extractor is not None else None

    # regions for separator features
    separator_regions = regions['SeparatorRegion'] if 'SeparatorRegion' in regions else None

    for i in range(num_interacting_nodes):
        edge_feature = []
        node_a, node_b = interacting_nodes[i, 0], interacting_nodes[i, 1]
        text_region_a, text_region_b = text_regions[node_a], text_regions[node_b]
        # separator feature (2-dim)
        if separator_regions:
            edge_feature.extend(get_edge_separator_feature(text_region_a, text_region_b, separator_regions))
        else:
            edge_feature.extend([0.0, 0.0])
        # text block similarity features based on word vectors
        if tb_sim_dict:
            try:
                edge_feature.extend(tb_sim_dict['edge_features'][text_region_a.id][text_region_b.id])
            except KeyError:
                edge_feature.extend(tb_sim_dict['edge_features']['default'])
        # external features
        for ext in external_data:
            try:
                ext_page = ext[os.path.basename(page_path)]
            except KeyError:
                logging.warning(f'Could not find key {page_path} in external data json.')
                continue
            if 'edge_features' in ext_page:
                try:
                    edge_feature.extend(ext['edge_features'][text_region_a.id][text_region_b.id])
                except (KeyError, TypeError):
                    try:
                        edge_feature.extend(ext['edge_features']['default'])
                    except KeyError:
                        edge_feature.extend([0.5])
        # final edge feature vector
        edge_features.append(edge_feature)

    # visual regions for nodes (for GNN visual features)
    visual_regions_nodes = []
    num_points_visual_regions_nodes = []
    if visual_regions:
        for text_region in text_regions:
            visual_regions_node = get_node_visual_regions(text_region)
            visual_regions_nodes.append(visual_regions_node)
            num_points_visual_regions_nodes.append(len(visual_regions_node))

    # visual regions for edges (for GNN visual features)
    visual_regions_edges = []
    num_points_visual_regions_edges = []
    if visual_regions:
        for i in range(num_interacting_nodes):
            node_a, node_b = interacting_nodes[i, 0], interacting_nodes[i, 1]
            text_region_a, text_region_b = text_regions[node_a], text_regions[node_b]
            visual_regions_edge = get_edge_visual_regions(text_region_a, text_region_b)
            visual_regions_edges.append(visual_regions_edge)
            num_points_visual_regions_edges.append(len(visual_regions_edge))

        # build padded array
        # make faster?
        # https://stackoverflow.com/questions/53071212/stacking-numpy-arrays-with-padding
        # https://stackoverflow.com/questions/53051560/stacking-numpy-arrays-of-different-length-using-padding/53052599?noredirect=1#comment93005810_53052599
        visual_regions_edges_array = np.zeros((num_interacting_nodes, np.max(num_points_visual_regions_edges), 2))
        for i in range(num_interacting_nodes):
            visual_region = visual_regions_edges[i]
            visual_regions_edges_array[i, :len(visual_region), :] = visual_region

    # ground-truth relations
    gt_relations = []
    # assign article_id to text_region based on most occuring text_lines
    # num_tr_uncertain = 0
    tr_gt_article_ids = []
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
            # num_tr_uncertain += 1
            assign_index = np.argmax(article_id_occurences)
            assign_article_id = unique_article_ids[int(assign_index)]
            tr_gt_article_ids.append(assign_article_id)
            # print(f"TextRegion {text_region.id}: assign article_id '{assign_article_id}' (from {unique_article_ids})")
        else:
            tr_gt_article_ids.append(unique_article_ids[0])
    # print(f"{num_tr_uncertain}/{len(text_regions)} text regions contained textlines of differing article_ids")
    # build gt ("1" means 'belong_to_same_article')
    for i, i_id in enumerate(tr_gt_article_ids):
        for j, j_id in enumerate(tr_gt_article_ids):
            if i_id == j_id:
                gt_relations.append([1, i, j])

    # number of ground-truth relations
    gt_num_relations = len(gt_relations)

    # TODO: transpose necessary?
    return np.array(num_nodes, dtype=np.int32), \
           interacting_nodes.astype(np.int32), \
           np.array(num_interacting_nodes, dtype=np.int32), \
           np.array(node_features, dtype=np.float32), \
           np.array(edge_features, dtype=np.float32) if edge_features else None, \
           np.transpose(np.array(visual_regions_nodes, dtype=np.float32), axes=(0, 2, 1)) if visual_regions else None, \
           np.array(num_points_visual_regions_nodes, dtype=np.int32) if visual_regions else None, \
           np.transpose(visual_regions_edges_array, axes=(0, 2, 1)) if visual_regions else None, \
           np.array(num_points_visual_regions_edges, dtype=np.int32) if visual_regions else None, \
           np.array(gt_relations, dtype=np.int32), \
           np.array(gt_num_relations, dtype=np.int32)


def get_data_from_pagexml(path_to_pagexml):
    """ Extraction of the information contained by a given Page xml file.

    :param path_to_pagexml: file path of the Page xml
    :return: list of polygons, list of article ids, image resolution
    """
    # load the page xml file
    page_file = Page(path_to_pagexml)

    # get text regions
    dict_of_regions = page_file.get_regions()

    # get all text lines of the loaded page file
    list_of_txt_lines = page_file.get_textlines()
    list_of_polygons = []
    list_of_article_ids = []
    for txt_line in list_of_txt_lines:
        try:
            # get the baseline of the text line as polygon
            list_of_polygons.append(txt_line.baseline.to_polygon())
            # get the article id of the text line
            list_of_article_ids.append(txt_line.get_article_id())
        except:
            print("\"NoneType\" object in PAGEXML with id {} has no attribute \"to_polygon\"!\n".format(txt_line.id))
            continue

    # image resolution
    resolution = page_file.get_image_resolution()
    return dict_of_regions, list_of_txt_lines, list_of_polygons, list_of_article_ids, resolution


# def generate_input_jsons_bc(page_list, json_list, out_path, num_node_features, num_edge_features,
#                             interaction, visual_regions):
def generate_input_jsons_bc(page_list, json_list, out_path,
                            interaction="delaunay",
                            visual_regions=True,
                            tb_similarity_setup=(None, None)):
    # Get page paths
    with open(page_list, "r") as page_files:
        page_paths = [os.path.abspath(line.rstrip()) for line in page_files.readlines()]
    assert all([file_name.endswith(".xml") for file_name in page_paths]), "Expected files to end in '.xml'"

    # Get json paths
    json_data = []
    for json_path in json_list:
        with open(json_path, "r") as json_file:
            json_data.append(json.load(json_file))

    # Setup textblock similarity feature extractor
    sim_feat_extractor = None
    if tb_similarity_setup[0] and tb_similarity_setup[1]:
        sim_feat_extractor = TextblockSimilarity(language=tb_similarity_setup[0], wv_path=tb_similarity_setup[1])

    # Get data from pagexml and write to json
    create_default_dir = False if out_path else True
    out_counter = 0
    for page_path in page_paths:
        # build input & target
        num_nodes, interacting_nodes, num_interacting_nodes, node_features, edge_features, \
        visual_regions_nodes, num_points_visual_regions_nodes, \
        visual_regions_edges, num_points_visual_regions_edges, \
        gt_relations, gt_num_relations = \
            build_input_and_target_bc(page_path=page_path,
                                      external_data=json_data,
                                      interaction=interaction,
                                      visual_regions=visual_regions,
                                      sim_feat_extractor=sim_feat_extractor)

        # build and write output
        if num_nodes is not None:
            out_dict = dict()
            out_dict["num_nodes"] = num_nodes.tolist()
            out_dict['interacting_nodes'] = interacting_nodes.tolist()
            out_dict['num_interacting_nodes'] = num_interacting_nodes.tolist()
            out_dict['node_features'] = node_features.tolist()
            out_dict['edge_features'] = edge_features.tolist()
            if visual_regions_nodes is not None and num_points_visual_regions_nodes is not None:
                out_dict['visual_regions_nodes'] = visual_regions_nodes.tolist()
                out_dict['num_points_visual_regions_nodes'] = num_points_visual_regions_nodes.tolist()
            if visual_regions_edges is not None and num_points_visual_regions_edges is not None:
                out_dict['visual_regions_edges'] = visual_regions_edges.tolist()
                out_dict['num_points_visual_regions_edges'] = num_points_visual_regions_edges.tolist()
            out_dict['gt_relations'] = gt_relations.tolist()
            out_dict['gt_num_relations'] = gt_num_relations.tolist()

            # Default output is a json folder one level above the pagexml file, indicating features and interaction
            if create_default_dir:
                visual = 'v' if visual_regions else ''
                out_path = re.sub(r'page$',
                                  f'json{node_features.shape[1]}{interaction[0]}{edge_features.shape[1]}{visual}',
                                  os.path.dirname(page_path))
            # Create output directory
            if not os.path.isdir(out_path):
                os.makedirs(out_path)
                print(f"Created output directory {out_path}")

            # Dump jsons
            file_name = os.path.splitext(file_name)[0] + ".json"
            out = os.path.join(out_path, file_name)
            with open(out, "w") as out_file:
                json.dump(out_dict, out_file)
                print(f"Wrote {out}")
                out_counter += 1
    print(f"Wrote {out_counter}/{len(page_paths)} files.")


if __name__ == '__main__':
    logging.getLogger().setLevel("WARN")

    # Register a custom function for 'bool' so --flag=True works.
    def str2bool(v):
        return v.lower() in ('true', 't', '1')

    parser = argparse.ArgumentParser()
    parser.add_argument('--pagexml_list', help="Input list with paths to pagexml files", required=True)
    parser.add_argument('--external_jsons', type=str, default=None, nargs='*',
                        help="External json files containing additional features")
    parser.add_argument('--out_dir', default="", help="Output directory for the json files")
    # parser.add_argument('--num_node_features', default=4, type=int,
    #                     help="Number of node features to be generated. Choose from ('4', '12', '14', '15', '16').")
    # parser.add_argument('--num_edge_features', default=0, type=int,
    #                     help="Number of edge features to be generated. Choose from ('0', '2').")
    parser.add_argument('--interaction', default='delaunay',
                        help="Determines the interacting nodes setup. Choose from ('fully', 'delaunay').")
    parser.add_argument('--visual_regions', nargs='?', const=True, default=False, type=str2bool,
                        help="Optionally build visual regions for nodes and edges (used for visual feature extraction)")
    parser.add_argument('--language', default=None, help="language used in tokenization and stopwords filtering "
                                                         "for text block similarites via word vectors")
    parser.add_argument('--wv_path', default=None, help="path to wordvector embeddings used for text block similarities")
    args = parser.parse_args()

    # generate_input_jsons_bc(args.pagexml_list, args.external_jsons, args.out_dir, args.num_node_features,
    #                         args.num_edge_features, args.interaction, args.visual_regions)
    generate_input_jsons_bc(args.pagexml_list,
                            args.external_jsons,
                            args.out_dir,
                            args.interaction,
                            args.visual_regions,
                            (args.language, args.wv_path))

    # page_path = "/home/johannes/devel/data/NewsEye_GT/AS_BC/NewsEye_ONB_232_textblocks/274950/ONB_nfp_18730705_corrected_duplicated/page/ONB_nfp_18730705_010.xml"
    # page_path = "/home/johannes/devel/data/NewsEye_GT/AS_BC/NewsEye_NLF_200_textblocks/330063/1869_01_04/page/ac-00001.xml"
    # page_path = "/home/johannes/devel/data/NewsEye_GT/AS_BC/NewsEye_ONB_232_textblocks/274954/ONB_ibn_19110701_corrected_duplicated/page/ONB_ibn_19110701_009.xml"

    # num_nodes, interacting_nodes, num_interacting_nodes, node_features, edge_features, \
    # visual_regions_nodes, num_points_visual_regions_nodes, visual_regions_edges, num_points_visual_regions_edges, \
    # gt_relations, gt_num_relations = \
    #     build_input_and_target_bc(page_path=page_path,
    #                               num_node_features=16,
    #                               num_edge_features=1,
    #                               interaction='delaunay',
    #                               visual_regions=True)

    # from input_fn.input_fn_rel.input_fn_generator_relation import get_img_from_page_path
    # from trainer.lav_types.eval_rel_bc import build_weighted_relation_graph, create_undirected_graph, plot_graph_and_page
    # page = Page(page_path)
    # img_path = get_img_from_page_path(page_path)
    # graph = build_weighted_relation_graph(interacting_nodes,
    #                                       [0.0 for i in range(len(interacting_nodes))],
    #                                       [{'separated_h': bool(e[0]), 'separated_v': bool(e[1])} for e in edge_features[:, :2]])
    # graph = create_undirected_graph(graph, weight_handling='avg', reciprocal=False)
    #
    # edge_colors = []
    # for u, v, d in graph.edges(data=True):
    #     color = 'g'
    #     if d['separated_v']:
    #         color = 'm'
    #     if d['separated_h']:
    #         color = 'r'
    #     edge_colors.append(color)
    # edge_cmap = None

    # import matplotlib.pyplot as plt
    # edge_colors = np.arange(len(list(graph.edges())))
    # edge_cmap = plt.get_cmap('jet')

    # plot_graph_and_page(page_path, graph, node_features, save_dir="/home/johannes",
    #                     with_edges=True, with_labels=True, edge_color=edge_colors, edge_cmap=edge_cmap)

    # edge_colors = []
    # for u, v, d in graph_full.edges(data='weight'):
    #     edge_colors.append(d)
    # plot_graph_and_page(page_path, graph_full, node_features, self._flags.debug_dir,
    #                     with_edges=True, with_labels=True, desc='confidences',
    #                     edge_color=edge_colors, edge_cmap=plt.get_cmap('jet'),
    #                     edge_vmin=0.0, edge_vmax=1.0)
