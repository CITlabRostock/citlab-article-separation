import os
from argparse import ArgumentParser
from collections import defaultdict

import cv2
import jpype
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from citlab_python_util.basic.list_util import filter_by_attribute
from citlab_python_util.geometry.point import rescale_points
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import ortho_connect
from citlab_python_util.image_processing.morphology import apply_transform
from citlab_python_util.parser.xml.page import plot as page_plot
from citlab_python_util.parser.xml.page.page import Page, Points
from citlab_python_util.plot import colors
from matplotlib.collections import PolyCollection

from citlab_article_separation.util import get_article_rectangles


def plot_gt_data(img_path, surr_polys_dict, show=True):
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
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors='k', facecolors='k')
        else:
            ar_poly_collection = PolyCollection(surr_polys, closed=True, edgecolors=colors.COLORS[i],
                                                facecolors=colors.COLORS[i])
        ar_poly_collection.set_alpha(0.5)
        ax.add_collection(ar_poly_collection)

    if show:
        plt.show()


def plot_polys_binary(polygon_list, img=None, img_width=None, img_height=None, closed=True):
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
    :param closed: draw a closed polygon or not
    :type closed: bool
    :return: pillow image
    """
    if img is None:
        # create binary image
        assert type(img_width) == int and type(img_height) == int, f"img_width and img_height must be integers but got " \
            f"the following values instead: {img_width} and {img_height}."
        img = Image.new('1', (img_width, img_height))
    pdraw = ImageDraw.Draw(img)
    for poly in polygon_list:
        if closed:
            pdraw.polygon(poly, outline="white")
        else:
            pdraw.line(poly, fill="white", width=7)

    return img


def rescale_image(img=None):
    if img is None:
        print("You must provide an image in order to rescale it.")
        exit(1)
    pass


if __name__ == '__main__':
    jpype.startJVM(jpype.getDefaultJVMPath())
    parser = ArgumentParser()
    parser.add_argument('--path_to_xml_lst', default='', type=str,
                        help="path to the lst file containing the file paths of the PageXMLs.")
    parser.add_argument('--path_to_img_lst', default='', type=str,
                        help='path to the lst file containing the file paths of the images.')
    parser.add_argument('--scaling_factor', default=0.5, type=int,
                        help='how much the GT images will be down-sampled, defaults to 0.5.')
    parser.add_argument('--save_folder', default='', type=str,
                        help='path to the folder the GT is written to.')
    args = parser.parse_args()

    if args.path_to_xml_lst == '':
        raise ValueError(f'Please provide a path to the list of PageXML files.')

    if args.path_to_img_lst == '':
        raise ValueError(f'Please provide a path to the list of image files.')

    if args.save_folder == '':
        raise ValueError(f'Please provide a valid save folder name.')

    if not os.path.exists(os.path.join(args.save_folder, 'C3')):
        os.makedirs(os.path.join(args.save_folder, 'C3'))

    path_to_xml_lst = './tests/resources/test_run_gt_generation/onb_page.lst'
    path_to_img_lst = './tests/resources/test_run_gt_generation/onb_img.lst'
    savedir = './tests/resources/test_run_gt_generation/onb_gt_contour_test/'

    with open(args.path_to_xml_lst) as f:
        with open(args.path_to_img_lst) as g:
            for path_to_page_xml, path_to_img in zip(f.readlines(), g.readlines()):
                path_to_page_xml = path_to_page_xml.strip()
                path_to_img = path_to_img.strip()

                page_filename = os.path.basename(path_to_page_xml)
                newspaper_filename = os.path.splitext(page_filename)[0]

                # Same structure as for the baseline detection training
                article_gt_savefile_name = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT0.png")
                baseline_gt_savefile_name = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT1.png")
                other_gt_savefile_name = os.path.join(args.save_folder, "C3", newspaper_filename + "_GT2.png")
                downscaled_grey_image_savefile_name = os.path.join(args.save_folder, newspaper_filename + ".png")

                rotation_savefile_name = downscaled_grey_image_savefile_name + ".rot"

                # create rotation files
                # TODO: only generates files with '0's in it -> fix this
                with open(rotation_savefile_name, "w") as rot:
                    rot.write("0")

                if os.path.isfile(article_gt_savefile_name) and os.path.isfile(baseline_gt_savefile_name) \
                        and os.path.isfile(other_gt_savefile_name) and os.path.isfile(
                    downscaled_grey_image_savefile_name) and os.path.isfile(rotation_savefile_name):
                    print(
                        f"GT Files for PageXml {path_to_page_xml} already exist, skipping...")
                    continue

                page = Page(path_to_page_xml)

                # Get the article rectangles as a list of ArticleRectangle objects
                ars, img_height, img_width = get_article_rectangles(page)

                # resize the image to draw the border polygons (if available)
                img_height += 1
                img_width += 1

                # get the width and height of the rescaled image and add 1 pixel to the borders
                img_scaled_height = int(img_height * args.scaling_factor) + 1
                img_scaled_width = int(img_width * args.scaling_factor) + 1

                # Convert the list of article rectangles to a dictionary with the article ids as keys
                # and the corresponding list of rectangles as value
                ars_dict = filter_by_attribute(ars, "a_ids")

                # Convert the article rectangles to surrounding polygons
                surr_polys_dict = defaultdict(list)
                article_polygon_img = None
                baseline_polygon_img = None
                for a_id, ars_sub in ars_dict.items():
                    if a_id == 'blank':
                        continue
                    rs = [Rectangle(ar.x, ar.y, ar.width, ar.height) for ar in ars_sub]
                    surr_polys = ortho_connect(rs)

                    # rescale the surrounding polygons
                    surr_polys_scaled = []
                    for sp in surr_polys:
                        surr_polys_scaled.append(rescale_points(sp, args.scaling_factor))

                    # returns a pillow image
                    article_polygon_img = plot_polys_binary(surr_polys_scaled, article_polygon_img,
                                                            img_height=img_scaled_height, img_width=img_scaled_width)
                    surr_polys_dict[a_id] = surr_polys_scaled

                    # Add the baselines to the baseline GT image
                    baseline_polygon_img = plot_polys_binary(
                        [rescale_points(tl.baseline.points_list, args.scaling_factor) for ar in ars_sub for tl in
                         ar.textlines],
                        baseline_polygon_img, img_height=img_scaled_height, img_width=img_scaled_width, closed=False)

                # Also add the printspace to the GT data
                ps_coords = page.get_print_space_coords()
                ps_poly = Points(ps_coords).to_polygon()
                ps_rectangle = ps_poly.get_bounding_box()
                article_polygon_img = plot_polys_binary(
                    [rescale_points(ps_rectangle.get_vertices(), args.scaling_factor)], article_polygon_img)

                # Comment out if you want to see the image with the corresponding regions before saving
                # plot_gt_data(path_to_img, surr_polys_dict)

                # convert pillow image to numpy array to use it in opencv
                def convert_and_apply_dilation(img, mode='article'):
                    # other modes: baseline
                    img_np = img.convert('L')
                    img_np = np.array(img_np, np.uint8)

                    if mode == 'article':
                        img_np = apply_transform(img_np, transform_type='dilation', kernel_size=(20, 20),
                                                 kernel_type='rect',
                                                 iterations=1)
                        img_np = apply_transform(img_np, transform_type='erosion', kernel_size=(13, 13),
                                                 kernel_type='rect',
                                                 iterations=1)
                    # elif mode == 'baseline':
                    #     img_np = apply_transform(img_np, transform_type='dilation', kernel_size=(1, 1),
                    #                              kernel_type='rect',
                    #                              iterations=1)

                    return img_np


                article_polygon_img_np = convert_and_apply_dilation(article_polygon_img, mode='article')
                baseline_polygon_img_np = convert_and_apply_dilation(baseline_polygon_img, mode='baseline')

                # make sure to reduce the image width and height by one pixel
                article_polygon_img_np = article_polygon_img_np[:-1, :-1]
                baseline_polygon_img_np = baseline_polygon_img_np[:-1, :-1]

                # Create the GT for the 'other' channel: white image minus the gt for article bounds and baselines
                other_img_np = 255 * np.ones(article_polygon_img_np.shape, np.uint8)
                other_img_np -= article_polygon_img_np
                other_img_np -= baseline_polygon_img_np
                # assign value 0 to the pixels that belong to a baseline as well as to an article boundary
                other_img_np *= ((other_img_np == 0) + (other_img_np == 255))

                # save article polygon image
                cv2.imwrite(article_gt_savefile_name, article_polygon_img_np)
                print(f'Saved file {article_gt_savefile_name}')
                # polygon_img.show()

                # save baseline polygon image
                cv2.imwrite(baseline_gt_savefile_name, baseline_polygon_img_np)
                print(f'Saved file {baseline_gt_savefile_name}')

                # save other image
                cv2.imwrite(other_gt_savefile_name, other_img_np)
                print(f'Saved file {other_gt_savefile_name}')

                # save the grey image in the same resolution as the GT data
                grey_img = Image.open(path_to_img).convert('L')
                assert grey_img.size == (img_width - 1, img_height - 1), f"resolutions of images don't match but are" \
                    f"{grey_img.size} and ({img_width - 1, img_height - 1})"
                grey_img_np = cv2.resize(np.array(grey_img, np.uint8), None, fx=args.scaling_factor,
                                         fy=args.scaling_factor, interpolation=cv2.INTER_AREA)
                cv2.imwrite(downscaled_grey_image_savefile_name, grey_img_np)
                print(f'Saved file {downscaled_grey_image_savefile_name}')

    jpype.shutdownJVM()
