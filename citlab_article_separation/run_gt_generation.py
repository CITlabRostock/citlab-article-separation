from argparse import ArgumentParser
from collections import defaultdict

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from citlab_python_util.basic.list_util import filter_by_attribute
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import ortho_connect
from citlab_python_util.image_processing.morphology import apply_transform
from citlab_python_util.parser.xml.page import plot as page_plot
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.plot import colors
from matplotlib.collections import PolyCollection

from citlab_article_separation.util import get_article_rectangles


def plot_gt_data(img_path, surr_polys_dict):
    """ Plots the groundtruth data for the article separation network or saves it to the directory

    :param img_path: path to the image the groundtruth is produced for
    :param article_rect_dict: the ArticleRectangle dictionary
    :param img_width: the width of the image
    :param img_height: the height of the image
    :param kernel_size: the size of the dilation kernel
    :param savedir: the directory the data should be stored
    :return:
    """
    fig, ax = plt.subplots()
    page_plot.add_image(ax, img_path)

    for i, a_id in enumerate(surr_polys_dict):
        # add facecolors="None" if rectangles should not be filled
        surr_polys = surr_polys_dict[a_id]
        if a_id == "blank":
            print(surr_polys)
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors='k', facecolors='k')
        else:
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors=colors.COLORS[i],
                                                facecolors=colors.COLORS[i])
        ar_poly_collection.set_alpha(0.5)
        ax.add_collection(ar_poly_collection)

    plt.show()


def plot_polys_binary(polygon_list, img=None, img_width=None, img_height=None):
    """Adds a list of polygons `polygon_list` to a pillow image `img`. If `img` is None a new pillow image is generated
    with a width of `img_width` and a height of `img_height`.

    :param polygon_list: a list of polygons, each given by a list of (x,y) tuples
    :type polygon_list: list of (list of (int, int))
    :param img: the pillow image to draw the polygons on
    :type img: Image.Image
    :param img_width: the width of the newly created image
    :type img_width: int
    :param img_height: the height of the newly created image
    :type img_height: int
    :return: pillow image
    """
    if img is None:
        # create binary image
        assert type(img_width) == int and type(img_height) == int, f"img_width and img_height must be integers but got " \
            f"the following values instead: {img_width} and {img_height}."
        img = Image.new('1', (img_width, img_height))
    pdraw = ImageDraw.Draw(img)
    for poly in polygon_list:
        pdraw.polygon(poly, outline="white")

    return img


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_to_xml_lst', default='', type=str, metavar="STR",
                        help="path to the lst file containing the file paths of the PageXMLs.")
    args = parser.parse_args()

    # if args.path_to_xml_lst == '':
    #     raise ValueError(f'Please provide a path to the list of PageXML files.')

    path_to_xml_lst = './tests/resources/test_run_gt_generation/onb_page_test.lst'
    savedir = './tests/resources/test_run_gt_generation/'

    with open(path_to_xml_lst) as f:
        for path_to_page_xml in f.readlines():
            path_to_page_xml = path_to_page_xml.strip()
            page = Page(path_to_page_xml)

            # Get the article rectangles as a list of ArticleRectangle objects
            ars, img_height, img_width = get_article_rectangles(page)

            # resize the image to draw the border polygons (if available)
            img_height += 1
            img_width += 1

            # Convert the list of article rectangles to a dictionary with the article ids as keys
            # and the corresponding list of rectangles as value
            ars_dict = filter_by_attribute(ars, "a_ids")
            print(ars_dict.keys())

            # Convert the article rectangles to surrounding polygons
            surr_polys_dict = defaultdict(list)
            polygon_img = None
            for a_id, ars_sub in ars_dict.items():
                if a_id == 'blank':
                    continue
                rs = [Rectangle(ar.x, ar.y, ar.width, ar.height) for ar in ars_sub]
                surr_polys = ortho_connect(rs)

                # returns a pillow image
                polygon_img = plot_polys_binary(surr_polys, polygon_img, img_height=img_height, img_width=img_width)
                # surr_polys_dict[a_id].append(surr_polys)
                surr_polys_dict[a_id] = surr_polys

            plot_gt_data('/home/max/data/as/NewsEye_ONB_Data/136358/ONB_aze_18950706/ONB_aze_18950706_5.jpg',
                         surr_polys_dict)

            # convert pillow image to numpy array to use it in opencv
            polygon_img_np = polygon_img.convert('L')
            polygon_img_np = np.array(polygon_img_np, np.uint8)

            polygon_img_np = apply_transform(polygon_img_np, transform_type='dilation', kernel_size=(20, 20),
                                             kernel_type='rect', iterations=1)

            # convert back to pillow image to save it properly
            # also make sure to reduce the image width and height by one pixel
            polygon_img = Image.fromarray(polygon_img_np[:-1, :-1])

            # save image
            polygon_img.save(savedir + path_to_page_xml.split("/")[-1] + '.png')
            # polygon_img.show()
