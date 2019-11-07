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
        print(height, width)

        horizontal_profiles = np.sum(self.images['text_block'], axis=1) / 255
        vertical_profiles = np.sum(self.images['text_block'], axis=0) / 255

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
    white_sep = cv2.resize(white_sep, None, fx=0.5, fy=0.5)

    separator_image = cv2.resize(separator_image, None, fx=0.5, fy=0.5)
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

    # print(tb_pp.get_rotation_angle(tb_pp.original_image))
    # print(tb_pp.get_rotation_angle(tb_outline_image))
    # print(tb_pp.get_rotation_angle(tb_image))
    # print(tb_pp.get_rotation_angle(separator_image))

    # a = np.array([[255, 0, 0, 0], [0, 255, 0, 0], [0, 0, 255, 0], [0, 0, 0, 255]])
    # b = np.array([[255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0], [255, 0, 0, 0]])
    # # b = np.array([[1, 2, 3], [4, 5, 6]])
    # # c = np.array([[0.1, 0.2], [1.2, 1.3]])
    #
    # _, best_angle = TextBlockNetPostProcessor.get_rotation_angle(b, delta=0.1, limit=90)
    # print(round(best_angle, 4))
