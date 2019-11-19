import logging
import os

import cv2
import numpy as np
from scipy.ndimage import interpolation as inter

logger = logging.getLogger("TextBlockNetPostProcessor")
logging.basicConfig(level=logging.WARNING)


class TextBlockNetPostProcessor(object):
    def __init__(self, original_image, text_block_outline, text_block, separator):
        self.images = {'original_image': original_image, 'text_block_outline': text_block_outline,
                       'text_block': text_block, 'separator': separator}
        # self.original_image = original_image
        # self.text_block_outline = text_block_outline
        # self.text_block = text_block
        # self.separator = separator

        if not self.check_dimensions(*tuple(self.images.values())):
            raise RuntimeError("Image shapes don't match.")

    @staticmethod
    def binarize_net_output(image, threshold):
        return np.array((image > threshold), np.int32)

    @staticmethod
    def get_rotation_angle(image, delta=0.1, limit=2):
        def find_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            hist = np.sum(data, axis=1)
            score = np.sum((hist[1:] - hist[:-1]) ** 2)
            return hist, score

        angles = np.arange(-limit, limit + delta, delta)
        scores = []
        for angle in angles:
            hist, score = find_score(image, angle)
            scores.append(score)

        best_score = max(scores)
        best_angle = angles[scores.index(best_score)]
        return best_score, best_angle

    def get_best_rotation_angle(self):
        return self.get_rotation_angle(self.images['original_image'])[1]

    @staticmethod
    def check_dimensions(*images):
        return all(image.shape == images[0].shape for image in images)

    def rotate_images(self, angle):
        for img_name, img in self.images.items():
            self.images[img_name] = inter.rotate(img, angle, reshape=False, order=0)

    def get_separators(self, threshold_horizontal=0.1, threshold_vertical=0.5):

        height, width = self.images['text_block'].shape

        horizontal_profiles = np.sum(self.images['text_block'], axis=1) / 255
        vertical_profiles = np.sum(self.images['text_block'], axis=0) / 255

        # We check for '<', because we search for blackruns in the textblock netoutput!
        horizontal_separators = [(i, hp / width) for i, hp in enumerate(horizontal_profiles) if
                                 hp / width < threshold_horizontal]
        vertical_separators = [(i, vp / height) for i, vp in enumerate(vertical_profiles) if
                               vp / height < threshold_vertical]

        print(len(horizontal_separators))
        print(horizontal_separators)
        print(len(vertical_separators))
        print(vertical_separators)

        return horizontal_separators, vertical_separators


if __name__ == '__main__':
    path_to_image_folder = '/home/max/devel/projects/python/article_separation/data/test_post_processing/textblock/'
    path_to_orig_image = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004.jpg')
    path_to_tb_outline = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT0.jpg')
    path_to_tb = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT1.jpg')
    path_to_separator = os.path.join(path_to_image_folder, 'ONB_aze_19110701_004_OUT2.jpg')

    orig_image = cv2.imread(path_to_orig_image, cv2.IMREAD_UNCHANGED)
    tb_outline_image = cv2.imread(path_to_tb_outline, cv2.IMREAD_UNCHANGED)
    tb_image = cv2.imread(path_to_tb, cv2.IMREAD_UNCHANGED)
    separator_image = cv2.imread(path_to_separator, cv2.IMREAD_UNCHANGED)

    orig_image = cv2.resize(orig_image, None, fx=0.4, fy=0.4)
    orig_image_gb = cv2.GaussianBlur(orig_image, (5, 5), 0)
    _, orig_image_bin = cv2.threshold(orig_image_gb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tb_pp = TextBlockNetPostProcessor(orig_image_bin, tb_outline_image, tb_image, separator_image)
    rotation_angle = round(tb_pp.get_best_rotation_angle(), 4)
    tb_pp.rotate_images(rotation_angle)

    horizontal_profile_list, vertical_profile_list = tb_pp.get_separators()

    index_horizontal = [i for i, _ in horizontal_profile_list]
    index_vertical = [i for i, _ in vertical_profile_list]

    white_sep = np.zeros(orig_image.shape, dtype=np.uint8)
    white_sep[:, index_vertical] = 255
    white_sep[index_horizontal, :] = 255
    # white_sep = cv2.resize(white_sep, None, fx=0.5, fy=0.5)

    # separator_image = cv2.resize(separator_image, None, fx=0.5, fy=0.5)
    separator_image = np.array((separator_image > 0.2), np.uint8)
    print(separator_image, separator_image.dtype)
    separator_image *= 255

    print(separator_image, separator_image.dtype)
    print(white_sep, white_sep.dtype)

    add_condition = np.not_equal(white_sep, separator_image)
    black_white_separator = np.copy(white_sep)
    black_white_separator[add_condition] += separator_image[add_condition]

    kernel = np.ones((5, 5), np.uint8)
    black_white_separator = cv2.morphologyEx(black_white_separator, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('white separator', white_sep)
    cv2.imshow('black separator net', separator_image)
    cv2.imshow('black white separator', black_white_separator)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    vertical_profile = np.sum(black_white_separator, axis=0)
    horizontal_profile = np.sum(black_white_separator, axis=1)

    horizontal = [(i, hp / orig_image.shape[1] / 255) for i, hp in enumerate(horizontal_profile) if
                  hp / orig_image.shape[1] / 255 < 0.2]
    vertical = [(i, vp / orig_image.shape[0] / 255) for i, vp in enumerate(vertical_profile) if
                vp / orig_image.shape[0] / 255 < 0.2]

    horizontal_index = [i for i, _ in horizontal]
    vertical_index = [i for i, _ in vertical]

    print(horizontal_index)
    print(vertical_index)


    def convert_to_ranges(index_list):
        range_list = []
        skip = False
        for i in range(len(index_list) - 1):
            if not skip:
                begin = index_list[i]
            if index_list[i + 1] - index_list[i] < 3:
                skip = True
                continue
            skip = False
            end = index_list[i]
            range_list.append((begin, end))
        return range_list

    print(convert_to_ranges(horizontal_index))
    print(convert_to_ranges(vertical_index))

    # tb_image_binarized = np.array((tb_image > 0.8), np.uint8) * 255
    # print(tb_image_binarized)
    # # erosion_kernel = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    # erosion_kernel = np.ones([8, 8], dtype=np.uint8)
    # print(erosion_kernel)
    # tb_image_erosion = cv2.erode(tb_image_binarized, erosion_kernel, iterations=1)
    # tb_image_erosion = cv2.resize(tb_image_erosion, None, fx=0.4, fy=0.4)
    # print(tb_image_erosion)
    # cv2.imshow("erosion image textblock", tb_image_erosion)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # exit(1)
