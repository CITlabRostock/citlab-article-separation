# -*- coding: utf-8 -*-

import sys
import copy
import jpype
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from PIL import Image
from util.xmlformats import Page
from util.misc import norm_poly_dists
from util.geometry import Rectangle, ArticleRectangle

# sys.setrecursionlimit(999)


class RegionRectangle(Rectangle):

    def __init__(self, x=0, y=0, width=0, height=0, polygon_tuple_list=None):
        super().__init__(x=x, y=y, width=width, height=height)

        self.polygon_tuple_list = polygon_tuple_list

    def create_subregion_rectangles(self, list_of_region_rectangles=None, image_width=None):
        """ Recursion for the segmentation of the page

        :param list_of_region_rectangles: recursively extended
        :param image_width:
        :return: list of region rectangles
        """
        if list_of_region_rectangles is None:
            list_of_region_rectangles = []

        # creation of 4 subrectangles
        width1 = self.width // 2
        width2 = self.width - width1
        height1 = self.height // 2
        height2 = self.height - height1

        rectangle1 = Rectangle(x=self.x, y=self.y, width=width1, height=height1)
        rectangle2 = Rectangle(x=self.x + width1, y=self.y, width=width2, height=height1)
        rectangle3 = Rectangle(x=self.x, y=self.y + height1, width=width1, height=height2)
        rectangle4 = Rectangle(x=self.x + width1, y=self.y + height1, width=width2, height=height2)

        # determine the bounding boxes of the textlines lying in the single subregions
        polygon_tuples_of_rec1 = []
        polygon_tuples_of_rec2 = []
        polygon_tuples_of_rec3 = []
        polygon_tuples_of_rec4 = []

        for polygon_tuple in self.polygon_tuple_list:

            intersection = rectangle1.intersection(polygon_tuple[0].bounds)
            if intersection.width > 0 and intersection.height > 0:
                polygon_tuples_of_rec1.append(polygon_tuple)

            intersection = rectangle2.intersection(polygon_tuple[0].bounds)
            if intersection.width > 0 and intersection.height > 0:
                polygon_tuples_of_rec2.append(polygon_tuple)

            intersection = rectangle3.intersection(polygon_tuple[0].bounds)
            if intersection.width > 0 and intersection.height > 0:
                polygon_tuples_of_rec3.append(polygon_tuple)

            intersection = rectangle4.intersection(polygon_tuple[0].bounds)
            if intersection.width > 0 and intersection.height > 0:
                polygon_tuples_of_rec4.append(polygon_tuple)

        region_rectangle1 = RegionRectangle(x=self.x, y=self.y, width=width1, height=height1,
                                            polygon_tuple_list=polygon_tuples_of_rec1)
        region_rectangle2 = RegionRectangle(x=self.x + width1, y=self.y, width=width2, height=height1,
                                            polygon_tuple_list=polygon_tuples_of_rec2)
        region_rectangle3 = RegionRectangle(x=self.x, y=self.y + height1, width=width1, height=height2,
                                            polygon_tuple_list=polygon_tuples_of_rec3)
        region_rectangle4 = RegionRectangle(x=self.x + width1, y=self.y + height1, width=width2, height=height2,
                                            polygon_tuple_list=polygon_tuples_of_rec4)

        # RECURSION: run "create_subregion_rectangles" on rectangles containing bounding boxes of textlines belonging
        # to different articles ("None" and "Blank" class are mentioned also as single classes)
        for reg_rec in [region_rectangle1, region_rectangle2, region_rectangle3, region_rectangle4]:
            list_of_ids = [t[1] for t in reg_rec.polygon_tuple_list]

            if image_width is None:
                if len(set(list_of_ids)) > 1:
                    reg_rec.create_subregion_rectangles(list_of_region_rectangles=list_of_region_rectangles)
                else:
                    list_of_region_rectangles.append(reg_rec)
            else:
                if len(set(list_of_ids)) > 1 or reg_rec.width > int(1/20 * image_width):
                    reg_rec.create_subregion_rectangles(list_of_region_rectangles=list_of_region_rectangles,
                                                        image_width=image_width)
                else:
                    list_of_region_rectangles.append(reg_rec)

        return list_of_region_rectangles


def get_data_from_pagexml(path_to_pagexml):
    """ Returns a list with all textlines contained by the PAGE XML file and the print space.

    :param path_to_pagexml: file path of the PAGE XML
    :return: list of textlines and the print space coordinates of the PAGE XML
    """
    # load the PAGE XML file
    page_file = Page.Page(path_to_pagexml)

    # get all textlines of the loaded page file
    list_of_txt_lines = page_file.get_textlines()
    # get print space of the loaded page file
    print_space = page_file.get_print_space_coords()

    # return list_of_txt_lines, rectangle
    return list_of_txt_lines, print_space


def initialize_gt_generation(list_of_textlines, des_dist=5, max_d=50):
    """ Creates a list of tuples of the form: (polygon (with corresponding bounding boxes depending on the interline
    distances), article id of the polygon).

    :param list_of_textlines: list of textline objects
    :param des_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons
    :param max_d: maximum distance (measured in pixels) for the calculation of the interline distances
    :return: list of tuples of the form (polygon, polygon id)
    """
    list_of_polygons = []
    list_of_polygon_ids = []

    for txt_line in list_of_textlines:
        try:
            txt_line_polygon = txt_line.baseline.to_polygon()
        except(AttributeError):
            print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(txt_line.id))
            continue

        txt_line_polygon.calculate_bounds()

        # txt_line_id \in {"None", "a1", "a2", "a3", ...}
        txt_line_id = txt_line.get_article_id()

        list_of_polygons.append(txt_line_polygon)
        list_of_polygon_ids.append(txt_line_id)

    # calculation of the normed polygons (includes also the calculation of their bounding boxes)
    list_of_normed_polygons = norm_poly_dists(list_of_polygons, des_dist=des_dist)

    # call java code to calculate the interline distances
    java_object = jpype.JPackage("util.Java_Util").JavaClass()

    list_of_normed_polygon_java = []

    for poly in list_of_normed_polygons:
        list_of_normed_polygon_java.append(jpype.java.awt.Polygon(poly.x_points, poly.y_points, poly.n_points))

    list_of_interline_distances_java = java_object.calcInterlineDistances(list_of_normed_polygon_java, des_dist, max_d)
    list_of_interline_distances = list(list_of_interline_distances_java)

    # computation of the bounding boxes of the polygons depending on the interline distances
    intersection_bool = True
    interline_distance_factor = 1.0
    list_of_polygons_expansion = copy.deepcopy(list_of_polygons)

    while intersection_bool:
        intersection_bool = False

        for poly_idx, polygon in enumerate(list_of_polygons_expansion):
            # bounding rectangle moved up and down
            height_shift = int(interline_distance_factor * list_of_interline_distances[poly_idx])

            polygon.bounds.translate(dx=0, dy=-height_shift)
            polygon.bounds.height += int(1.1 * height_shift)

        # check whether there are polygons with different id's intersecting each other
        # this would lead to endless recursion!
        for i in range(len(list_of_polygons_expansion)):
            if intersection_bool:
                break

            for j in range(i + 1, len(list_of_polygons_expansion)):
                intersection = list_of_polygons_expansion[i].bounds.intersection(list_of_polygons_expansion[j].bounds)
                if intersection.width > 0 and intersection.height > 0 \
                        and list_of_polygon_ids[i] != list_of_polygon_ids[j]:

                    intersection_bool = True

                    list_of_polygons_expansion = copy.deepcopy(list_of_polygons)
                    if interline_distance_factor > 0.1:
                        interline_distance_factor -= 0.05
                    else:
                        interline_distance_factor *= 0.5

                    break

    print("used interline distance factor: {}\n".format(interline_distance_factor))

    return list(zip(list_of_polygons_expansion, list_of_polygon_ids))


def get_id_rectangle_dict(list_of_region_rectangles):
    """ Returns a dictionary with the article id's (also "None" and "Blank" are valid id's) and the corresponding lists
    of region rectangles.

    :param list_of_region_rectangles: list of region rectangle objects
    :return: dictionary of the form: {id: list of rectangles}
    """
    id_rectangle_dict = {}

    for reg_rec in list_of_region_rectangles:
        list_of_ids = [t[1] for t in reg_rec.polygon_tuple_list]

        if len(list_of_ids) == 0:
            if "Blank" in id_rectangle_dict:
                id_rectangle_dict["Blank"].append(reg_rec)
            else:
                id_rectangle_dict.update({"Blank": [reg_rec]})
        elif list_of_ids[0] is None:
            if "None" in id_rectangle_dict:
                id_rectangle_dict["None"].append(reg_rec)
            else:
                id_rectangle_dict.update({"None": [reg_rec]})
        else:
            if list_of_ids[0] in id_rectangle_dict:
                id_rectangle_dict[list_of_ids[0]].append(reg_rec)
            else:
                id_rectangle_dict.update({list_of_ids[0]: [reg_rec]})

    return id_rectangle_dict


def plot_region_rectangles(image_file, id_rectangle_dict):
    colors = ["darkgreen", "red", "darkviolet", "darkblue",
              "gold", "darkorange", "brown", "yellowgreen", "darkcyan",

              "darkkhaki", "firebrick", "darkorchid", "deepskyblue",
              "peru", "orangered", "rosybrown", "burlywood", "cadetblue",

              "olivedrab", "palevioletred", "plum", "slateblue",
              "tan", "coral", "sienna", "yellow", "mediumaquamarine",

              "forestgreen", "indianred", "blueviolet", "steelblue",
              "silver", "salmon", "darkgoldenrod", "greenyellow", "darkturquoise",

              "mediumseagreen", "crimson", "rebeccapurple", "navy",
              "darkgray", "saddlebrown", "maroon", "lawngreen", "royalblue",

              "springgreen", "tomato", "violet", "azure",
              "goldenrod", "chocolate", "chartreuse", "teal"]

    colors_dict = {}

    for i in range(len(colors)):
        if i == 0:
            colors_dict.update({"None": "black", "Blank": "gray"})
        else:
            colors_dict.update({"a" + str(i): colors[i - 1]})

    im = np.array(Image.open(image_file))
    # create figure and axes
    fig, ax = plt.subplots()

    # display the image
    ax.imshow(im)

    for article_id in id_rectangle_dict:
        for rectangle in id_rectangle_dict[article_id]:
            # create a rectangle patch
            rect = patches.Rectangle((rectangle.x, rectangle.y), rectangle.width, rectangle.height, alpha=0.75,
                                     linewidth=1, edgecolor=colors_dict[article_id], facecolor=colors_dict[article_id])
            # add the patch to the axes
            ax.add_patch(rect)

    plt.title(image_file.split("/")[-1])
    plt.show()


if __name__ == "__main__":
    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    # page_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/xml_paths_aze.lst", "r").readlines()
    # image_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/image_paths_aze.lst", "r").readlines()
    #
    page_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/xml_paths_ibn.lst", "r").readlines()
    image_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/image_paths_ibn.lst", "r").readlines()
    #
    # page_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/xml_paths_krz.lst", "r").readlines()
    # image_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/image_paths_krz.lst", "r").readlines()
    #
    # page_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/xml_paths_nfp.lst", "r").readlines()
    # image_paths = open("/home/basti/Documents/Job_Rostock/NewsEye/data/image_paths_nfp.lst", "r").readlines()

    page_file_list = [path.strip("\n\r") for path in page_paths]
    image_file_list = [path.strip("\n\r") for path in image_paths]

    for index, page_file in enumerate(page_file_list):
        image_file = image_file_list[index]

        print("\nPAGE file: {}".format(page_file))
        print("Image file: {}".format(image_file))

        txt_lines, print_space_rectangle = get_data_from_pagexml(page_file)

        width_image = print_space_rectangle[1][0] - print_space_rectangle[0][0]
        height_image = print_space_rectangle[2][1] - print_space_rectangle[0][1]

        list_polygon_tuple = initialize_gt_generation(list_of_textlines=txt_lines, max_d=int(1/20 * height_image))

        regrec = RegionRectangle(x=print_space_rectangle[0][0], y=print_space_rectangle[0][1],
                                 width=print_space_rectangle[1][0] - print_space_rectangle[0][0],
                                 height=print_space_rectangle[2][1] - print_space_rectangle[0][1],
                                 polygon_tuple_list=list_polygon_tuple)

        list_region_rectangles = \
            regrec.create_subregion_rectangles(list_of_region_rectangles=None, image_width=width_image)

        dict_id_rectangle = get_id_rectangle_dict(list_of_region_rectangles=list_region_rectangles)

        plot_region_rectangles(image_file=image_file, id_rectangle_dict=dict_id_rectangle)


        #############
        # old version
        #
        # gt_object = ArticleRectangle(print_space_rectangle[0][0], print_space_rectangle[0][1],
        #                              print_space_rectangle[1][0] - print_space_rectangle[0][0],
        #                              print_space_rectangle[2][1] - print_space_rectangle[0][1], txt_lines)
        #
        # list_of_region_rectangles = gt_object.create_subregions()
        #
        # id_rectangle_dict = {}
        #
        # for reg_rec in list_of_region_rectangles:
        #     list_of_ids = list(reg_rec.a_ids)
        #
        #     if len(list_of_ids) == 0:
        #         if "Blank" in id_rectangle_dict:
        #             id_rectangle_dict["Blank"].append(reg_rec)
        #         else:
        #             id_rectangle_dict.update({"Blank": [reg_rec]})
        #     elif list_of_ids[0] is None:
        #         if "None" in id_rectangle_dict:
        #             id_rectangle_dict["None"].append(reg_rec)
        #         else:
        #             id_rectangle_dict.update({"None": [reg_rec]})
        #     else:
        #         if list_of_ids[0] in id_rectangle_dict:
        #             id_rectangle_dict[list_of_ids[0]].append(reg_rec)
        #         else:
        #             id_rectangle_dict.update({list_of_ids[0]: [reg_rec]})
        #
        # plot_region_rectangles(image_file=image_file, id_rectangle_dict=id_rectangle_dict)

    # shut down the java virtual machine
    jpype.shutdownJVM()
