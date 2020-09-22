from citlab_article_separation.net_post_processing.region_net_post_processor_base import RegionNetPostProcessor
from citlab_article_separation.net_post_processing.text_region_to_page_writer import TextRegionToPageWriter
from citlab_python_util.logging.custom_logging import setup_custom_logger
from citlab_python_util.parser.xml.page.page_constants import sTEXTREGION

logger = setup_custom_logger("TextBlockNetPostProcessor", "info")


class TextBlockNetPostProcessor(RegionNetPostProcessor):
    def __init__(self, image_list, path_to_pb, fixed_height, scaling_factor, threshold):
        super().__init__(image_list, path_to_pb, fixed_height, scaling_factor, threshold)

    def to_page_xml(self, page_path, image_path=None, polygons_dict=None, *args, **kwargs):
        region_page_writer = TextRegionToPageWriter(page_path, image_path, self.fixed_height, self.scaling_factor,
                                                    polygons_dict)
        region_page_writer.overwrite_text_regions()

        logger.debug(f"Saving TextBlockNetPostProcessor results to page {page_path}")
        # region_page_writer.save_page_xml(page_path + ".xml")

        return region_page_writer.page_object

    def post_process(self, net_output):
        # net_output = self.apply_morphology_operators(net_output)
        net_output = net_output[:, :, 0]
        net_output_post = self.apply_cc_analysis(net_output, 1 / net_output.size * 100)

        return net_output_post

    def to_polygons(self, net_output):
        contours = self.apply_contour_detection(net_output, use_alpha_shape=False)
        contours = [self.remove_every_nth_point(contour, n=2, min_num_points=20, iterations=1) for contour in
                    contours]
        return {sTEXTREGION: contours}


if __name__ == '__main__':
    # ONB Test Set (3000 height)
    image_list = "/home/max/data/la/textblock_detection/newseye_tb_data/onb/tmp.lst"
    # image_list = "/home/max/data/la/racetrack_onb_corrected_baselines/racetrack_onb_corrected_baselines.lst"
    # Independance_lux dataset (3000 height)
    # image_list = '/home/max/data/la/textblock_detection/bnl_data/independance_lux/traindata_headers/val.lst'

    # Textblock detection
    path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
                 "racetrack_onb_textblock_136/TB_aru_3000_height_scaling_train_only/export/" \
                 "TB_aru_3000_height_scaling_train_only_2020-06-05.pb"
    path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/heading_detection/HD_aru_3000_export_best/export/HD_aru_3000_export_best_2020-09-22.pb"
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/experiments_for_gnns/TB_ru_016_050/export/TB_ru_016_050_2020-09-02.pb"

    # Header detection
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/independance_lux/headers/" \
    #              "tb_headers_aru/export/tb_headers_aru_2020-06-04.pb"
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/textblock_detection/newseye/" \
    #              "racetrack_onb_textblock_136/with_headings/TB_aru_3000_height/export/TB_aru_3000_height_2020-06-10.pb"

    # Separators
    # path_to_pb = "/home/max/devel/projects/python/aip_pixlab/models/separator_detection/SEP_aru_5300/export/" \
    #              "SEP_aru_5300_2020-06-10.pb"
    tb_pp = TextBlockNetPostProcessor(image_list, path_to_pb, fixed_height=None, scaling_factor=1.0, threshold=0.05)
    # tb_pp = TextBlockNetPostProcessor2(path_to_pb, fixed_height=None, scaling_factor=0.7, threshold=0.1)
    tb_pp.run()
    # for i, image in enumerate(tb_pp.images):
    #     tb_pp.plot_binary(image, tb_pp.net_outputs[i][0, :, :, 0])
