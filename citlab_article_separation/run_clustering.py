# -*- coding: utf-8 -*-

import jpype
from argparse import ArgumentParser

from citlab_python_util.parser.xml.page.page import Page
from citlab_article_separation import dbscan_baselines


def get_data_from_pagexml(path_to_pagexml):
    """

    :param path_to_pagexml: file path
    :return: a list of tuples (text of text line, baseline of text line)
    """
    # load the page xml file
    page_file = Page(path_to_pagexml)
    # get all text lines of the loaded page file
    list_of_txt_lines = page_file.get_textlines()

    list_of_text = []
    list_of_polygons = []

    for txt_line in list_of_txt_lines:
        try:
            # get the baseline of the text line as polygon
            list_of_polygons.append(txt_line.baseline.to_polygon())
            # get the text of the text line
            list_of_text.append(txt_line.text)
        except(AttributeError):
            print("'NoneType' object in PAGEXML with id {} has no attribute 'to_polygon'!\n".format(txt_line.id))
            continue

    list_of_tuples = [(list_of_text[i], list_of_polygons[i]) for i in range(len(list_of_text))]
    # image resolution
    resolution = page_file.get_image_resolution()

    return list_of_tuples, resolution


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


def cluster_baselines_dbscan(data, min_polygons_for_cluster=1, des_dist=5, max_d=50, min_polygons_for_article=2,
                             rectangle_interline_factor=2, bounding_box_epsilon=10,
                             use_java_code=True):
    """

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
    :return: list with article labels for each data tuple (i.e. for each text line)
    """
    # initialization of the clustering algorithm object
    cluster_object \
        = dbscan_baselines.DBSCANBaselines \
        (data, min_polygons_for_cluster=min_polygons_for_cluster, des_dist=des_dist, max_d=max_d,
         min_polygons_for_article=min_polygons_for_article,
         rectangle_interline_factor=rectangle_interline_factor, bounding_box_epsilon=bounding_box_epsilon,
         use_java_code=use_java_code)

    # AS algorithm based on DBSCAN
    cluster_object.clustering_polygons()
    article_list = cluster_object.get_cluster_of_polygons()

    return article_list


if __name__ == "__main__":
    parser = ArgumentParser()
    # command-line argument
    parser.add_argument('--path_to_xml_lst', default='', type=str, metavar="STR",
                        help="path to the lst file containing the file paths of the page xml's to be processed")

    # start java virtual machine to be able to execute the java code
    jpype.startJVM(jpype.getDefaultJVMPath())

    # example with command-line argument
    flags = parser.parse_args()
    hypo_files_paths_list = flags.path_to_xml_lst

    # hypo_files_paths_list = "/home/basti/Documents/Job_Rostock/NewsEye/HYPO_DATA/onb_nfp_19110701_wordvec_test/xml_paths.lst"
    # hypo_files_paths_list = "/home/basti/Documents/Job_Rostock/NewsEye/HYPO_DATA/bnf_set_to_cluster_5912/xml_paths.lst"

    hypo_files = [line.rstrip('\n') for line in open(hypo_files_paths_list, "r")]
    skipped_files = []

    for counter, hypo_file in enumerate(hypo_files):
        print(hypo_file)

        data, image_resolution = get_data_from_pagexml(hypo_file)
        max_d = int(1 / 50 * image_resolution[1])

        article_id_list = cluster_baselines_dbscan(data, min_polygons_for_cluster=1, min_polygons_for_article=2,
                                                   max_d=max_d, bounding_box_epsilon=10, rectangle_interline_factor=2)
        try:
            save_results_in_pagexml(hypo_file, article_id_list)
        except:
            print("Can not save the results of the clustering in the Page xml: ", hypo_file)
            skipped_files.append(hypo_file)

        print("Progress: {:.2f} %".format(((counter + 1) / len(hypo_files)) * 100))

    print("\nNumber of skipped pages since storing errors: ", len(skipped_files))
    print(skipped_files)

    # shut down the java virtual machine
    jpype.shutdownJVM()
