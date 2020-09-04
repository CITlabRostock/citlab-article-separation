import argparse

from citlab_article_separation.net_post_processing.region_net_post_processor_base import RegionNetPostProcessor
from citlab_article_separation.net_post_processing.separator_region_to_page_writer import SeparatorRegionToPageWriter
from citlab_python_util.parser.xml.page.page_constants import sSEPARATORREGION


class SeparatorNetPostProcessor(RegionNetPostProcessor):

    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, threshold):
        super().__init__(image_list, path_to_pb, fixed_height, scaling_factor, threshold)

    def post_process(self, net_output):
        """
        Post process the raw net output `net_output`, e.g. by removing CCs that have a size smaller than 100 pixels.
        :param net_output: numpy array with dimension HWC (only channel 1 - the separator channel - is important)
        :return: post processed net output of the same dimension as the input (HWC)
        """
        # Ignore the other class
        net_output = net_output[:, :, 0]
        # Delete connected components that have a size of less than 100 pixels
        net_output_post = self.apply_cc_analysis(net_output, 1 / net_output.size * 100)

        return net_output_post

    def to_polygons(self, net_output):
        """
        Converts the (post-processed) net output `net_output` to a list of polygons via contour detection and removal of
        unnecessary points.
        :param net_output: numpy array with dimension HWC
        :return: list of polygons representing the contours of the CCs is appended to the `net_output_polygons`
        attribute
        """
        # for net_output in self.net_outputs_post:
        contours = self.apply_contour_detection(net_output, use_alpha_shape=False)
        # contours = [self.remove_every_nth_point(contour, n=2, min_num_points=20, iterations=2) for contour in
        #             contours]

        return {sSEPARATORREGION: contours}

    def to_page_xml(self, page_path, polygons_dict, image_path=None):
        """
        Write the polygon information given by `polygons_dict` coming from the `to_polygons` function to the page file
        in `page_path`.
        :param page_path: path to the page file the region information should be written to
        :param polygons_dict: dictionary with region types as keys and the corresponding list of polygons as values
        :return: Page object that can either be further processed or be written to
        """
        # Load the region-to-page-writer and initalize it with the given page path and its region dictionary
        region_page_writer = SeparatorRegionToPageWriter(page_path, polygons_dict, image_path, self.fixed_height,
                                                         self.scaling_factor)
        region_page_writer.remove_separator_regions_from_page()
        region_page_writer.merge_regions()
        region_page_writer.save_page_xml(page_path + ".xml")

        return region_page_writer.page_object


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str, required=True,
                        help="Path to the image list for which separator information should be created.")
    parser.add_argument('--path_to_pb', type=str, required=True,
                        help="Path to the TensorFlow pb graph for creating the separator information")
    parser.add_argument('--fixed_height', type=int, required=False,
                        help="If parameter is given, the images will be scaled to this height by keeping the aspect "
                             "ratio", default=None)
    parser.add_argument('--scaling_factor', type=float, required=False,
                        help="If no --fixed_height flag is given, use a predefined scaling factor on the images.",
                        default=1.0)
    parser.add_argument('--threshold', type=float, required=False,
                        help="Threshold value that is used to convert the probability outputs of the neural network"
                             "to 0 and 1 values", default=0.05)

    args = parser.parse_args()

    image_list = args.image_list
    path_to_pb = args.path_to_pb
    fixed_height = args.fixed_height
    scaling_factor = args.scaling_factor
    threshold = args.threshold

    post_processor = SeparatorNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, threshold)
    post_processor.run()

    # # ONB Test Set (3000 height)
    # # image_list = "/home/max/data/la/textblock_detection/newseye_tb_data/onb/tmp.lst"
    # # Independance_lux dataset (3000 height)
    # # image_list = '/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst'
    # image_list = "/home/max/sanomia_turusta_544852/images.lst"
    #
    # # Textblock detection
    # path_to_pb_tb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                 "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
    #                 "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"
    #
    # # Header detection
    # # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/headers/" \
    # #              "tb_headers_aru/export/tb_headers_aru_2020-06-04.pb"
    # path_to_pb_hd = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #                 "racetrack_onb_textblock_136/with_headings/TB_aru_3000_height/export/TB_aru_3000_height_2020-06-10.pb"
    #
    # # Separators
    # path_to_pb_sp = "/home/max/devel/projects/python/aip_pixlab/models/separator_detection/SEP_aru_5300/export/" \
    #                 "SEP_aru_5300_2020-06-10.pb"
    #
    # # Comparison dbscan vs pixellabeling
    # # image_list = "/home/max/sanomia_turusta_544852/images.lst"
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_tb, fixed_height=1650, scaling_factor=1.0, threshold=0.2)
    #
    # image_list = "/home/max/separator_splits_with_words/images.lst"
    #
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_tb, fixed_height=None, scaling_factor=0.55, threshold=0.2)
    # # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_hd, fixed_height=None, scaling_factor=0.55, threshold=0.2)
    # tb_pp = SeparatorNetPostProcessor(image_list, path_to_pb_sp, fixed_height=None, scaling_factor=1.0, threshold=0.05)
    # tb_pp.run()
