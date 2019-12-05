# -*- coding: utf-8 -*-

import jpype
from argparse import ArgumentParser

from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation import dbscan_baselines


def get_data_from_pagexml(path_to_pagexml):
    """
    :param path_to_pagexml: file path
    :return: list of polygons
    """
    # load the page xml file
    page_file = Page(path_to_pagexml)
    # get all text lines of the loaded page file
    list_of_txt_lines = page_file.get_textlines()

    list_of_polygons = []

    for txt_line in list_of_txt_lines:
        try:
            # get the baseline of the text line as polygon
            list_of_polygons.append(txt_line.baseline.to_polygon())
        except(AttributeError):
            print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(txt_line.id))
            continue

    return list_of_polygons


def save_results_in_pagexml(path_to_pagexml, list_of_txt_line_labels):
    """
    :param path_to_pagexml: file path
    :param list_of_txt_line_labels: list of article tags of the baselines
    """
    page_file = Page(path_to_pagexml)
    # get all text lines of the loaded page file
    list_of_txt_lines = page_file.get_textlines()

    txt_line_index = -1

    for i, txt_line in enumerate(list_of_txt_lines):
        txt_line_index += 1

        try:
            txt_line.baseline.to_polygon()
        except(AttributeError):
            txt_line_index -= 1
            continue

        # existing article tags are overwritten!
        if list_of_txt_line_labels[txt_line_index] == -1:
            txt_article_id = txt_line.get_article_id()

            if txt_article_id is not None:
                txt_line.set_article_id(article_id=None)
        else:
            txt_line.set_article_id(article_id="a" + str(list_of_txt_line_labels[txt_line_index]))

    page_file.set_textline_attr(list_of_txt_lines)
    page_file.write_page_xml(path_to_pagexml)


def cluster_baselines_dbscan(list_of_polygons, min_polygons_for_cluster=1,  min_polygons_for_article=2,
                             bounding_box_epsilon=5, rectangle_interline_factor=2,
                             des_dist=5, max_d=500, use_java_code=True, target_average_interline_distance=0):
    """
    :param list_of_polygons: list_of_polygons
    :param min_polygons_for_cluster: minimum number of required polygons in neighborhood to form a cluster
    :param min_polygons_for_article: minimum number of required polygons forming an article

    :param bounding_box_epsilon: additional width and height value to calculate the bounding boxes of the polygons
                                 during the clustering progress
    :param rectangle_interline_factor: multiplication factor to calculate the height of the rectangles during the
                                       clustering progress with the help of the interline distances

    :param des_dist: desired distance (measured in pixels) of two adjacent pixels in the normed polygons
    :param max_d: maximum distance (measured in pixels) for the calculation of the interline distances
    :param use_java_code: usage of methods written in java (faster than python!) or not
    :param target_average_interline_distance: target interline distance for scaling of the polygons

    :return: list with article labels for each data tuple (i.e. for each text line)
    """
    # initialization of the clustering algorithm object
    cluster_object \
        = dbscan_baselines.DBSCANBaselines \
        (list_of_polygons=list_of_polygons,
         min_polygons_for_cluster=min_polygons_for_cluster, min_polygons_for_article=min_polygons_for_article,
         bounding_box_epsilon=bounding_box_epsilon, rectangle_interline_factor=rectangle_interline_factor,
         des_dist=des_dist, max_d=max_d, use_java_code=use_java_code,
         target_average_interline_distance=target_average_interline_distance)

    # AS algorithm based on DBSCAN
    cluster_object.clustering_polygons()
    article_list = cluster_object.get_cluster_of_polygons()

    return article_list


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line arguments
    parser.add_argument('--path_to_xml_lst', type=str, required=True,
                        help="path to the lst file containing the file paths of the page xml's to be processed")

    parser.add_argument('--min_polygons_for_cluster', type=int, default=1,
                        help="minimum number of required polygons in neighborhood to form a cluster")
    parser.add_argument('--min_polygons_for_article', type=int, default=2,
                        help="minimum number of required polygons forming an article")

    parser.add_argument('--bounding_box_epsilon', type=int, default=5,
                        help="additional width and height value to calculate the bounding boxes of the polygons during "
                             "the clustering progress")
    parser.add_argument('--rectangle_interline_factor', type=float, default=2,
                        help="multiplication factor to calculate the height of the rectangles during the clustering "
                             "progress with the help of the interline distances")

    parser.add_argument('--des_dist', type=int, default=5,
                        help="desired distance (measured in pixels) of two adjacent pixels in the normed polygons")
    parser.add_argument('--max_d', type=int, default=500,
                        help="maximum distance (measured in pixels) for the calculation of the interline distances")
    parser.add_argument('--use_java_code', type=bool, default=True,
                        help="usage of methods written in java (faster than python!) or not")
    parser.add_argument('--target_average_interline_distance', type=int, default=50,
                        help="target interline distance for scaling of the polygons")

    flags = parser.parse_args()

    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    xml_files = [line.rstrip('\n') for line in open(flags.path_to_xml_lst, "r")]
    skipped_files = []

    for i, xml_file in enumerate(xml_files):
        print(xml_file)
        article_id_list = cluster_baselines_dbscan(list_of_polygons=get_data_from_pagexml(path_to_pagexml=xml_file),
                                                   min_polygons_for_cluster=flags.min_polygons_for_cluster,
                                                   min_polygons_for_article=flags.min_polygons_for_article,
                                                   bounding_box_epsilon=flags.bounding_box_epsilon,
                                                   rectangle_interline_factor=flags.rectangle_interline_factor,
                                                   des_dist=flags.des_dist, max_d=flags.max_d,
                                                   use_java_code=flags.use_java_code,
                                                   target_average_interline_distance=flags.target_average_interline_distance)
        try:
            save_results_in_pagexml(xml_file, article_id_list)
        except:
            print("Can not save the results of the clustering in the Page xml: ", xml_file)
            skipped_files.append(xml_file)

        print("\nProgress: {:.2f} %\n".format(((i + 1) / len(xml_files)) * 100))

    print("\nNumber of skipped pages since storing errors: ", len(skipped_files))
    print(skipped_files)

    # shut down the java virtual machine
    jpype.shutdownJVM()
