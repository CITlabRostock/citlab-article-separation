import logging
import os
from abc import ABC, abstractmethod
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image, ImageDraw
from citlab_python_util.geometry.point import rescale_points
from citlab_python_util.parser.xml.page.page import Page

logger = logging.getLogger("GroundTruthGenerator")


def check_if_files_exist(*filenames):
    files_exist = map(os.path.isfile, filenames)
    return all(files_exist)


def load_image_list(path_to_img_lst):
    with open(path_to_img_lst, "r") as f:
        ret = f.readlines()
    return [img_path.rstrip() for img_path in ret]


class GroundTruthGenerator(ABC):
    def __init__(self, path_to_img_lst, fixed_height=0, scaling_factor=1.0):
        self.img_path_lst = load_image_list(path_to_img_lst)
        self.page_path_lst = self.get_page_list()
        self.page_object_lst = self.create_page_objects()
        self.img_res_lst_original = self.get_image_resolutions_from_page_objects()  # list of tuples (width, height) containing the size of the image
        self.fixed_height = max(0, fixed_height)
        if self.fixed_height:
            self.scaling_factors = self.calculate_scaling_factors_from_fixed_height()
        else:
            sc_factor = max(0.1, scaling_factor)
            self.scaling_factors = [sc_factor] * len(self.img_path_lst)
        self.grey_img_lst, self.img_res_lst = self.create_grey_images()
        self.gt_imgs_lst = []  # list of tuples
        self.gt_polygon_lst = []  # list of tuples representing lists of polygons plotted in gt_imgs_lst
        self.n_channels = 0
        super().__init__()

    def create_page_objects(self):
        return [Page(page_path) for page_path in self.page_path_lst]

    def create_grey_images(self):
        new_img_res_lst = []
        grey_img_lst = []
        for i, path_to_img in enumerate(self.img_path_lst):
            grey_img = Image.open(path_to_img).convert('L')
            grey_img = np.array(grey_img, np.uint8)

            if self.scaling_factors[i] < 1:
                grey_img = cv2.resize(grey_img, None, fx=self.scaling_factors[i], fy=self.scaling_factors[i],
                                      interpolation=cv2.INTER_AREA)
            elif self.scaling_factors[i] > 1:
                grey_img = cv2.resize(grey_img, None, fx=self.scaling_factors[i], fy=self.scaling_factors[i],
                                      interpolation=cv2.INTER_CUBIC)

            new_img_res_lst.append(grey_img.shape)
            grey_img_lst.append(grey_img)

        return grey_img_lst, new_img_res_lst

    @abstractmethod
    def create_ground_truth_images(self):
        pass

    def get_page_list(self):
        page_path_lst = []
        for img_path in self.img_path_lst:
            dirname = os.path.dirname(img_path)
            img_filename = os.path.basename(img_path)
            page_filename = os.path.splitext(img_filename)[0] + '.xml'
            page_path_lst.append(os.path.join(dirname, "page", page_filename))
        return page_path_lst

    def save_ground_truth(self, savedir):
        if not self.gt_imgs_lst:
            print("No ground truth images to save.")
        else:
            if not os.path.isdir(savedir) or not os.path.exists(savedir):
                os.mkdir(savedir)

            for i, gt_imgs in enumerate(self.gt_imgs_lst):
                for j, gt_img in enumerate(gt_imgs):
                    gt_img_savefile_name = self.get_ground_truth_image_savefile_name(self.img_path_lst[i], j, savedir,
                                                                                     gt_folder_name="C" + str(len(gt_imgs)))
                    cv2.imwrite(gt_img_savefile_name, gt_img)
                cv2.imwrite(self.get_grey_image_savefile_name(self.img_path_lst[i], savedir), self.grey_img_lst[i])
                with open(self.get_rotation_savefile_name(self.img_path_lst[i], savedir), "w") as rot:
                    rot.write("0")

    @staticmethod
    def create_other_ground_truth_image(*channel_images):
        other_img_np = 255 * np.ones(channel_images[0].shape, np.uint8)

        for channel_img in channel_images:
            other_img_np -= channel_img

        other_img_np *= ((other_img_np == 0) + (other_img_np == 255))

        return other_img_np

    @staticmethod
    def get_ground_truth_image_savefile_name(img_name, index, save_dir, gt_folder_name="C3", gt_file_ext=".png"):
        channel_gt_dir = os.path.join(save_dir, gt_folder_name)
        if not os.path.exists(channel_gt_dir) or not os.path.isdir(channel_gt_dir):
            os.mkdir(channel_gt_dir)
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, gt_folder_name, img_name_wo_ext + "_GT" + str(index) + gt_file_ext)

    @staticmethod
    def get_grey_image_savefile_name(img_name, save_dir, grey_img_file_ext=".jpg"):
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, img_name_wo_ext + grey_img_file_ext)

    @staticmethod
    def get_rotation_savefile_name(img_name, save_dir, rotation_file_ext=".jpg.rot"):
        img_name_wo_ext = os.path.splitext(os.path.basename(img_name))[0]
        return os.path.join(save_dir, img_name_wo_ext + rotation_file_ext)

    def run_ground_truth_generation(self, save_dir):
        self.create_grey_images()
        self.create_page_objects()
        self.create_ground_truth_images()

        self.save_ground_truth(save_dir)

    @staticmethod
    def rescale_polygon(polygon, scaling_factor):
        return rescale_points(polygon, scaling_factor) if scaling_factor else polygon

    @staticmethod
    def plot_polys_binary(polygon_list, img=None, img_width=None, img_height=None, closed=True, fill_polygons=False,
                          line_width=7):
        if img is None:
            # create binary image
            assert type(img_width) == int and type(
                img_height) == int, f"img_width and img_height must be integers but got " \
                                    f"the following values instead: {img_width} and {img_height}."
            img = Image.new('1', (img_width, img_height))
        pdraw = ImageDraw.Draw(img)
        for poly in polygon_list:
            if closed:
                if fill_polygons:
                    pdraw.polygon(poly, outline="white", fill="white")
                else:
                    poly_closed = deepcopy(poly)
                    poly_closed.append(poly[0])
                    # pdraw.polygon(poly, outline="white")
                    pdraw.line(poly_closed, fill="white", width=line_width)
            else:
                pdraw.line(poly, fill="white", width=line_width)

        img = img.convert('L')
        return np.array(img, np.uint8)

    @staticmethod
    def make_disjoint(gt_img_compare, gt_img_to_change):
        # assume both images are numpy arrays of the same size
        return np.where(gt_img_compare == gt_img_to_change, 0, gt_img_to_change)

    def make_disjoint_all(self):
        for i, gt_imgs in enumerate(self.gt_imgs_lst):
            changed_gt_imgs = [gt_imgs[0]]
            for j in range(len(gt_imgs) - 1):
                changed_gt_imgs.append(self.make_disjoint(gt_img_compare=gt_imgs[j], gt_img_to_change=gt_imgs[j + 1]))
            self.gt_imgs_lst[i] = tuple(changed_gt_imgs)

    def get_image_resolutions_from_page_objects(self):
        return [page.get_image_resolution() for page in self.page_object_lst]

    def calculate_scaling_factors_from_fixed_height(self):
        if not self.fixed_height:
            logger.warning("No fixed height given, do nothing...")
        return [self.fixed_height / img_res[1] for img_res in self.img_res_lst_original]
