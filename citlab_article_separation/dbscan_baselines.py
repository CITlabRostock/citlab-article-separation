# -*- coding: utf-8 -*-
""" DBSCAN based on Chris McCormicks "https://github.com/chrisjmccormick/dbscan" """

import math
import jpype
import collections

from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import calc_reg_line_stats, get_dist_fast, get_in_dist, get_off_dist
from citlab_python_util.geometry.polygon import norm_poly_dists


class DBSCANBaselines:

    def __init__(self, data, min_polygons_for_cluster=2, des_dist=5, max_d=50, min_polygons_for_article=3,
                 rectangle_interline_factor=3 / 2,
                 bounding_box_epsilon=5,
                 use_java_code=True):
        """ Initialization of the clustering process.

        :param data: list of tuples ("String", Polygon) as the dataset
        :param min_polygons_for_cluster: minimum number of required polygons forming a cluster
        :param des_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons
        :param max_d: maximum distance (measured in pixels) for the calculation of the interline distances
        :param min_polygons_for_article: minimum number of required polygons forming an article

        :param rectangle_interline_factor: multiplication factor to calculate the height of the rectangles with the help
                                           of the interline distances
        :param bounding_box_epsilon: additional width and height value to calculate the bounding boxes of the polygons
                                     during the clustering progress

        :param use_java_code: usage of methods written in java or not
        """
        self.data = data
        self.min_polygons_for_cluster = min_polygons_for_cluster
        self.max_d = max_d
        self.min_polygons_for_article = min_polygons_for_article

        self.rectangle_interline_factor = rectangle_interline_factor
        self.bounding_box_epsilon = bounding_box_epsilon
        # self.min_intersect_ratio = min_intersect_ratio

        list_of_polygons = [tpl[1] for tpl in self.data]
        # calculation of the normed polygons (includes also the calculation of their bounding boxes)
        self.list_of_normed_polygons = norm_poly_dists(list_of_polygons, des_dist=des_dist)

        # call java code to calculate the interline distances
        if use_java_code:
            java_object = jpype.JPackage("citlab_article_separation.java").Util()

            list_of_nomred_polygon_java = []

            for poly in self.list_of_normed_polygons:
                list_of_nomred_polygon_java.append(jpype.java.awt.Polygon(poly.x_points, poly.y_points, poly.n_points))

            list_of_interline_distances_java = \
                java_object.calcInterlineDistances(list_of_nomred_polygon_java, des_dist, self.max_d)

            self.list_of_interline_distances = list(list_of_interline_distances_java)

        # call python code to calculate the interline distances
        else:
            self.list_of_interline_distances = []
            # calculation of the interline distances for all normed polygons
            self.list_of_interline_distances = \
                DBSCANBaselines.calc_interline_dist(self, tick_dist=des_dist, max_d=self.max_d)

        # initially all labels for all baselines are 0 (0 means the baseline hasn't been considered yet,
        # -1 stands for noise, clusters are numbered starting from 1)
        self.list_of_labels = [0] * len(self.list_of_normed_polygons)
        self.list_if_center = [False] * len(self.list_of_normed_polygons)

        print("Number of (detected) baselines contained by the image: {}".format(len(self.list_of_normed_polygons)))

    def clustering_polygons(self):
        """ Clusters the polygons with DBSCAN based approach. """
        label = 0

        # if valid center polygon is found, a new cluster is created
        for polygon_index in range(len(self.list_of_normed_polygons)):
            # if the polygon's label isn't 0, continue to the next polygon
            if not (self.list_of_labels[polygon_index] == 0):
                continue

            # find all neighboring polygons
            neighbor_polygons = DBSCANBaselines.region_query(self, polygon_index)

            # if the number is below "min_polygons_for_cluster", this polygon is "noise"
            # a noise polygon may later be picked up by another cluster as a boundary polygon
            if len(neighbor_polygons) < self.min_polygons_for_cluster:
                self.list_of_labels[polygon_index] = -1

            # otherwise, this polygon is a center polygon for a new cluster
            else:
                label += 1
                self.list_if_center[polygon_index] = True

                # build the cluster
                DBSCANBaselines.grow_cluster(self, polygon_index, neighbor_polygons, label)

    def grow_cluster(self, polygon_index, neighbor_polygons, this_label):
        """ Grows a new cluster with label "this_label" from a center polygon with index "polygon_index".

        :param polygon_index: index of a center polygon of this new cluster
        :param neighbor_polygons: all neighbors of the center polygon
        :param this_label: label for this new cluster
        """
        # assign the cluster label to the center polygon
        self.list_of_labels[polygon_index] = this_label

        # look at each neighbor of the center polygon; neighbor polygons will be used as a FIFO queue (while-loop)
        # of polygons to search - it will grow as we discover new polygons for the cluster

        i = 0
        while i < len(neighbor_polygons):
            neighbor_index = neighbor_polygons[i]

            # if "neighbor_index" was labelled noise, we know it's not a center polygon (not enough neighbors),
            # so make it a boundary polygon of the cluster and move on
            if self.list_of_labels[neighbor_index] == -1:
                self.list_of_labels[neighbor_index] = this_label

            # if "neighbor_index" isn't already claimed, add the polygon to the cluster
            elif self.list_of_labels[neighbor_index] == 0:
                self.list_of_labels[neighbor_index] = this_label

                # find all the neighbors of "neighbor_index"
                next_neighbor_polygons = DBSCANBaselines.region_query(self, neighbor_index)

                # if "neighbor_index" has at least "min_polygons_for_cluster" neighbors, it's a new center polygon
                # add all of its neighbors to the FIFO queue
                if len(next_neighbor_polygons) >= self.min_polygons_for_cluster:
                    self.list_if_center[neighbor_index] = True
                    neighbor_polygons += next_neighbor_polygons

                # if "neighbor_index" doesn't have enough neighbors, don't queue up it's neighbors as expansion polygons

            # if another center polygon is in our neighborhood, merge the two clusters
            elif self.list_of_labels[neighbor_index] != this_label and self.list_if_center[neighbor_index]:
                self.list_of_labels = \
                    [self.list_of_labels[neighbor_index] if x == this_label else x for x in self.list_of_labels]

                this_label = self.list_of_labels[neighbor_index]

            # next point in the FIFO queue
            i += 1

    def region_query(self, polygon_index):
        """ Finds all polygons in the dataset within the defined neighborhood of the considered polygon "polygon_index".

        :param polygon_index: index of the considered polygon
        :return: index list of the neighbor polygons
        """
        neighbors = []

        for i, normed_polygon_i in enumerate(self.list_of_normed_polygons):
            if i == polygon_index:
                continue

            bool_inter = DBSCANBaselines.neighborhood(self, polygon1_index=polygon_index, polygon2_index=i)
            if bool_inter:
                neighbors.append(i)

        return neighbors

    def neighborhood(self, polygon1_index, polygon2_index):
        """ Decides, whether two given polygons "polygon1_index" and "polygon2_index" lie within a defined neighborhood.

        :param polygon1_index: index of the first polygon
        :param polygon2_index: index of the second polygon
        :return: True or False
        """
        eps = self.bounding_box_epsilon
        fac = self.rectangle_interline_factor
        average_interline_distance = 1 / len(self.list_of_interline_distances) * sum(self.list_of_interline_distances)

        # computation of two different rectangles for polygon 1
        poly1 = self.list_of_normed_polygons[polygon1_index]
        # int_dis1 = self.list_of_interline_distances[polygon1_index]
        int_dis1 = average_interline_distance

        if poly1.bounds.width > 2 * eps:
            rec1 = Rectangle(int(poly1.bounds.x + eps), int(poly1.bounds.y - eps),
                             int(poly1.bounds.width - 2 * eps), int(poly1.bounds.height + 2 * eps))
        else:
            rec1 = Rectangle(int(poly1.bounds.x), int(poly1.bounds.y - eps),
                             int(poly1.bounds.width), int(poly1.bounds.height + 2 * eps))

        rec1_expanded = Rectangle(int(poly1.bounds.x - eps), int(poly1.bounds.y - fac * int_dis1),
                                  int(poly1.bounds.width + 2 * eps), int(poly1.bounds.height + 2 * fac * int_dis1))

        # computation of two different rectangles for polygon 2
        poly2 = self.list_of_normed_polygons[polygon2_index]
        # int_dis2 = self.list_of_interline_distances[polygon2_index]
        int_dis2 = average_interline_distance

        if poly2.bounds.width > 2 * eps:
            rec2 = Rectangle(int(poly2.bounds.x + eps), int(poly2.bounds.y - eps),
                             int(poly2.bounds.width - 2 * eps), int(poly2.bounds.height + 2 * eps))
        else:
            rec2 = Rectangle(int(poly2.bounds.x), int(poly2.bounds.y - eps),
                             int(poly2.bounds.width), int(poly2.bounds.height + 2 * eps))

        rec2_expanded = Rectangle(int(poly2.bounds.x - eps), int(poly2.bounds.y - fac * int_dis2),
                                  int(poly2.bounds.width + 2 * eps), int(poly2.bounds.height + 2 * fac * int_dis2))

        # computation of intersection rectangles
        intersection_1to2 = rec1_expanded.intersection(rec2)
        intersection_2to1 = rec2_expanded.intersection(rec1)

        # computation of intersection surfaces
        if intersection_1to2.width > 0 and intersection_1to2.height > 0:
            intersection1to2_surface = intersection_1to2.width * intersection_1to2.height
        else:
            intersection1to2_surface = 0

        if intersection_2to1.width > 0 and intersection_2to1.height > 0:
            intersection2to1_surface = intersection_2to1.width * intersection_2to1.height
        else:
            intersection2to1_surface = 0

        # computation of rectangle surfaces
        rec1_surface = rec1.height * rec1.width
        rec2_surface = rec2.height * rec2.width

        if intersection1to2_surface >= rec2_surface or intersection2to1_surface >= rec1_surface:
            return True

        return False

    def get_cluster_of_polygons(self):
        """ Calculates the cluster labels for the polygons.

        :return: list with article labels for each polygon
        """
        # articles with less than "min_polygons_for_article" polygons belong to the "noise" class
        counter_dict = collections.Counter(self.list_of_labels)

        for label in counter_dict:
            if counter_dict[label] < self.min_polygons_for_article and label != -1:
                self.list_of_labels = [-1 if x == label else x for x in self.list_of_labels]

        counter_dict = collections.Counter(self.list_of_labels)
        print("Number of detected articles (inclusive the \"noise\" class): {}\n".format(len(counter_dict)))

        return self.list_of_labels

    def calc_interline_dist(self, tick_dist=5, max_d=250):
        """ Calculates interline distance values for every (normed!) polygon according to
            "https://arxiv.org/pdf/1705.03311.pdf".

        :param tick_dist: desired distance of points of the polygon
        :param max_d: max distance of pixels of a polygon to any other polygon (distance in terms of the x- and y-
                      distance of the point to a bounding box of another polygon - see get_dist_fast)
        :return: interline distances for every polygon
        """
        interline_dist = []

        for poly_a in self.list_of_normed_polygons:
            # calculate the angle of the linear regression line representing the baseline polygon poly_a
            angle = calc_reg_line_stats(poly_a)[0]

            # orientation vector (given by angle) of length 1
            or_vec_y, or_vec_x = math.sin(angle), math.cos(angle)
            dist = max_d

            # first and last point of polygon
            pt_a1 = [poly_a.x_points[0], poly_a.y_points[0]]
            pt_a2 = [poly_a.x_points[-1], poly_a.y_points[-1]]

            # iterate over pixels of the current GT baseline polygon
            for x_a, y_a in zip(poly_a.x_points, poly_a.y_points):
                p_a = [x_a, y_a]
                # iterate over all other polygons (to calculate X_G)
                for poly_b in self.list_of_normed_polygons:
                    if poly_b != poly_a:
                        # if polygon poly_b is too far away from pixel p_a, skip
                        if get_dist_fast(p_a, poly_b.get_bounding_box()) > dist:
                            continue

                        # get first and last pixel of baseline polygon poly_b
                        pt_b1 = poly_b.x_points[0], poly_b.y_points[0]
                        pt_b2 = poly_b.x_points[-1], poly_b.y_points[-1]

                        # calculate the inline distance of the points
                        in_dist1 = get_in_dist(pt_a1, pt_b1, or_vec_x, or_vec_y)
                        in_dist2 = get_in_dist(pt_a1, pt_b2, or_vec_x, or_vec_y)
                        in_dist3 = get_in_dist(pt_a2, pt_b1, or_vec_x, or_vec_y)
                        in_dist4 = get_in_dist(pt_a2, pt_b2, or_vec_x, or_vec_y)
                        if (in_dist1 < 0 and in_dist2 < 0 and in_dist3 < 0 and in_dist4 < 0) or (
                                in_dist1 > 0 and in_dist2 > 0 and in_dist3 > 0 and in_dist4 > 0):
                            continue

                        for p_b in zip(poly_b.x_points, poly_b.y_points):
                            if abs(get_in_dist(p_a, p_b, or_vec_x, or_vec_y)) <= 2 * tick_dist:
                                dist = min(dist, abs(get_off_dist(p_a, p_b, or_vec_x, or_vec_y)))

            if dist < max_d:
                interline_dist.append(dist)
            else:
                interline_dist.append(max_d)

        return interline_dist
