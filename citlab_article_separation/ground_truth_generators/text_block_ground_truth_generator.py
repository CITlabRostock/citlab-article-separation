import argparse
import logging

import cv2
import numpy as np
from citlab_python_util.parser.xml.page import page_constants

from citlab_article_separation.ground_truth_generators.ground_truth_generator_base import GroundTruthGenerator

logger = logging.getLogger("TextBlockGroundTruthGenerator")
logging.basicConfig(level=logging.WARNING)


class TextBlockGroundTruthGenerator(GroundTruthGenerator):
    def __init__(self, path_to_img_lst, fixed_height=0, scaling_factor=1.0, use_bounding_box=False,
                 use_min_area_rect=False):
        super().__init__(path_to_img_lst, fixed_height, scaling_factor)
        self.create_page_objects()
        self.regions_list = [page.get_regions() for page in self.page_object_lst]
        self.image_regions_list = self.get_image_regions_list()
        self.separator_regions_list = self.get_separator_regions_list()
        self.text_regions_list = self.get_valid_text_regions()
        self.use_bounding_box = use_bounding_box
        self.use_min_area_rect = use_min_area_rect

    def create_ground_truth_images(self):
        # Textblocks outlines, Textblocks filled, (Images), Separators, Other
        for i in range(len(self.img_path_lst)):
            # Textblock outline GT image
            img_width = self.img_res_lst[i][1]
            img_height = self.img_res_lst[i][0]
            sc_factor = self.scaling_factors[i]

            # tb_outlines_gt_img = self.create_region_gt_img(self.text_regions_list[i], img_width, img_height, fill=False,
            #                                                scaling_factor=sc_factor)
            tb_filled_gt_img = self.create_region_gt_img(self.text_regions_list[i], img_width, img_height, fill=True,
                                                         scaling_factor=sc_factor)
            # image_region_gt_img = self.create_region_gt_img(self.image_regions_list[i], img_width, img_height, fill=True,
            #                                                 scaling_factor=sc_factor)
            # sep_region_gt_img = self.create_region_gt_img(self.separator_regions_list[i], img_width, img_height,
            #                                               fill=True, scaling_factor=sc_factor)
            gt_channels = [tb_filled_gt_img]
            # gt_channels = [tb_outlines_gt_img, tb_filled_gt_img, sep_region_gt_img]
            # gt_channels = [tb_outlines_gt_img, tb_filled_gt_img, image_region_gt_img, sep_region_gt_img]

            other_gt_img = self.create_other_ground_truth_image(*gt_channels)
            gt_channels.append(other_gt_img)
            gt_channels = tuple(gt_channels)

            self.gt_imgs_lst.append(gt_channels)
        self.make_disjoint_all()

    def get_min_area_rect(self, points):
        points_np = np.array(points)
        min_area_rect = cv2.minAreaRect(points_np)
        min_area_rect = cv2.boxPoints(min_area_rect)
        min_area_rect = np.int0(min_area_rect)

        min_area_rect = min_area_rect.tolist()
        min_area_rect = [tuple(p) for p in min_area_rect]

        return min_area_rect

    def create_region_gt_img(self, regions, img_width: int, img_height: int, fill: bool, scaling_factor: float = None):
        if self.use_bounding_box:
            regions_polygons = [region.points.to_polygon().get_bounding_box().get_vertices() for region in regions]
        elif self.use_min_area_rect:
            regions_polygons = [self.get_min_area_rect(region.points.to_polygon().as_list()) for region in regions]
        else:
            regions_polygons = [region.points.to_polygon().as_list() for region in regions]

        region_gt_img = self.plot_polys_binary([self.rescale_polygon(rp, scaling_factor) for rp in regions_polygons],
                                               img_width=img_width, img_height=img_height, fill_polygons=fill,
                                               closed=True)
        return region_gt_img

    def get_valid_text_regions(self, intersection_thresh=20, region_type='paragraph'):
        valid_text_regions_list = []
        for i, regions in enumerate(self.regions_list):
            valid_text_regions = []
            text_regions = [region for region in regions["TextRegion"] if region.region_type == region_type]
            image_regions = self.image_regions_list[i]
            if not image_regions:
                valid_text_regions_list.append(text_regions)
                continue

            text_regions_bbs = [text_region.points.to_polygon().get_bounding_box() for text_region in text_regions]
            image_regions_bbs = [image_region.points.to_polygon().get_bounding_box() for image_region in image_regions]

            for j, text_region_bb in enumerate(text_regions_bbs):
                for image_region_bb in image_regions_bbs:
                    if image_region_bb.contains_rectangle(text_region_bb):
                        break
                    intersection = text_region_bb.intersection(image_region_bb)
                    if intersection.height > intersection_thresh and intersection.width > intersection_thresh:
                        break
                else:
                    valid_text_regions.append(text_regions[j])

            valid_text_regions_list.append(valid_text_regions)

        return valid_text_regions_list

    def get_table_regions_list(self):
        return self.get_regions_list([page_constants.sTABLEREGION])

    def get_advert_regions_list(self):
        return self.get_regions_list([page_constants.sADVERTREGION])

    def get_image_regions_list(self):
        return self.get_regions_list([page_constants.sGRAPHICREGION, page_constants.sIMAGEREGION])

    def get_separator_regions_list(self):
        return self.get_regions_list([page_constants.sSEPARATORREGION])

    def get_regions_list(self, region_types):
        region_list_by_type = []
        for i, page_regions in enumerate(self.regions_list):
            regions = []
            for region_type in region_types:
                try:
                    regions += page_regions[region_type]
                except KeyError:
                    logger.debug("No {} for PAGE {}.".format(region_type, self.page_path_lst[i]))
                region_list_by_type.append(regions)

        return region_list_by_type

    def get_title_regions_list(self, title_region_types):
        """ Valid title_region_types are ["headline", "subheadline", "publishing_stmt", "motto", "other" ]
        """

        return self.get_heading_regions_list('title', title_region_types)

    def get_classic_heading_regions_list(self, heading_region_types):
        """ Valid class_heading_region_types are ["overline", "", "subheadline", "author", "other"]
        where "" represents the title.
        """
        return self.get_heading_regions_list('heading', heading_region_types)

    def get_heading_regions_list(self, custom_structure_type, custom_structure_subtypes):
        valid_text_regions = self.get_valid_text_regions(region_type=page_constants.TextRegionTypes.sHEADING)

        region_list_by_type = []
        if len(valid_text_regions) == 0:
            region_list_by_type.append([])
        for page_text_regions in valid_text_regions:
            regions = []
            for page_text_region in page_text_regions:
                custom_dict_struct = page_text_region.custom['structure']
                for custom_struct_subtype in custom_structure_subtypes:
                    if custom_struct_subtype == '' and custom_dict_struct['type'] == custom_structure_type and 'subtype' not in custom_dict_struct.keys():
                        regions.append(page_text_region)
                    elif custom_dict_struct['type'] == custom_structure_type and custom_dict_struct['subtype'] == custom_struct_subtype:
                        regions.append(page_text_region)
            region_list_by_type.append(regions)

        return region_list_by_type


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_list', type=str)
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--fixed_height', type=int, default=0)
    parser.add_argument('--scaling_factor', type=float, default=1.0)

    args = parser.parse_args()

    tb_generator = TextBlockGroundTruthGenerator(
        args.image_list, use_bounding_box=False, use_min_area_rect=False, fixed_height=args.fixed_height,
        scaling_factor=args.scaling_factor)
    # print(tb_generator.image_regions_list)
    # print(tb_generator.text_regions_list)
    # tb_generator.create_grey_images()
    # tb_generator.create_page_objects()
    # img_height = tb_generator.img_res_lst[0][0]
    # img_width = tb_generator.img_res_lst[0][1]
    # tb_gt = tb_generator.create_region_gt_img(tb_generator.text_regions_list[0], img_width, img_height, fill=True)
    # tb_surr_poly_gt = tb_generator.create_region_gt_img(tb_generator.text_regions_list[0], img_width, img_height,
    #                                                     fill=False)
    # cv2.imwrite("./data/tb_gt.png", tb_gt)
    # cv2.imwrite("./data/tb_surr_poly_gt.png", tb_surr_poly_gt)

    # imgplot = plt.imshow(gt_img, cmap="gray")
    # plt.show()
    # cv2.imshow("test", gt_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    tb_generator.run_ground_truth_generation(args.save_dir)
