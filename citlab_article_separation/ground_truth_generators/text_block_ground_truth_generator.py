import logging

import cv2
import numpy as np

from citlab_article_separation.ground_truth_generators.ground_truth_generator_base import GroundTruthGenerator

logger = logging.getLogger("TextBlockGroundTruthGenerator")
logging.basicConfig(level=logging.WARNING)


class TextBlockGroundTruthGenerator(GroundTruthGenerator):
    def __init__(self, path_to_img_lst, fixed_height=0, scaling_factor=1.0, use_bounding_box=False, use_min_area_rect=False):
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
            # tb_filled_gt_img = self.create_region_gt_img(self.text_regions_list[i], img_width, img_height, fill=True,
            #                                              scaling_factor=sc_factor)
            # image_region_gt_img = self.create_region_gt_img(self.image_regions_list[i], img_width, img_height, fill=True,
            #                                                 scaling_factor=sc_factor)
            sep_region_gt_img = self.create_region_gt_img(self.separator_regions_list[i], img_width, img_height,
                                                          fill=True, scaling_factor=sc_factor)
            gt_channels = [sep_region_gt_img]
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

    def get_valid_text_regions(self, intersection_thresh=20):
        valid_text_regions_list = []
        for i, regions in enumerate(self.regions_list):
            valid_text_regions = []
            text_regions = regions["TextRegion"]
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

    def get_image_regions_list(self):
        image_regions_list = []
        for i, regions in enumerate(self.regions_list):
            image_regions = []
            try:
                image_regions += regions["GraphicRegion"]
            except KeyError:
                logger.debug(f"No GraphicRegion for PAGE '{self.page_path_lst[i]}' found.")
            try:
                image_regions += regions["ImageRegions"]
            except KeyError:
                logger.debug(f"No ImageRegion for PAGE '{self.page_path_lst[i]}' found.")

            image_regions_list.append(image_regions)

        return image_regions_list

    def get_separator_regions_list(self):
        return [regions["SeparatorRegion"] for regions in self.regions_list]


if __name__ == '__main__':
    tb_generator = TextBlockGroundTruthGenerator(
        '/home/max/devel/projects/python/article_separation/lists/textblock_detection/onb_newspaper_tb.lst',
        use_bounding_box=False, use_min_area_rect=False, fixed_height=5300)
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

    tb_generator.run_ground_truth_generation('./data/rot_test/')
