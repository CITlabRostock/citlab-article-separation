import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from PIL import Image, ImageDraw

from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_objects import *
from citlab_python_util.parser.xml.page import plot
from citlab_python_util.geometry.rectangle import Rectangle
from citlab_article_separation.article_rectangle import ArticleRectangle

COLORS = ["darkgreen", "red", "darkviolet", "darkblue",
          "gold", "darkorange", "brown", "yellowgreen", "darkcyan",

          "darkkhaki", "firebrick", "darkorchid", "deepskyblue",
          "peru", "orangered", "rosybrown", "burlywood", "cadetblue",

          "olivedrab", "palevioletred", "plum", "slateblue",
          "tan", "coral", "sienna", "yellow", "mediumaquamarine",

          "forestgreen", "indianred", "blueviolet", "steelblue",
          "silver", "salmon", "darkgoldenrod", "greenyellow", "darkturquoise",

          "mediumseagreen", "crimson", "rebeccapurple", "navy",
          "darkgray", "saddlebrown", "maroon", "lawngreen", "royalblue",

          "springgreen", "tomato", "violet", "azure",
          "goldenrod", "chocolate", "chartreuse", "teal"]


def get_article_rectangles(page):
    """Given the PageXml file `page` return the corresponding article subregions as a list of ArticleRectangle objects.
     Also returns the width and height of the image (NOT of the PrintSpace).

    :param page: Either the path to the PageXml file or a Page object.
    :type page: Union[str, Page]
    :return: the article subregion list, the width and height of the image
    """
    if type(page) == str:
        page = Page(page)

    assert type(page) == Page, f"Type must be Page, got {type(page)} instead."
    ps_coords = page.get_print_space_coords()
    ps_poly = Points(ps_coords).to_polygon()
    # Maybe check if the surrounding Rectangle of the polygon has corners given by ps_poly
    ps_rectangle = ps_poly.get_bounding_box()

    # First ArticleRectangle to consider
    ps_rectangle = ArticleRectangle(ps_rectangle.x, ps_rectangle.y, ps_rectangle.width, ps_rectangle.height,
                                    page.get_textlines())

    ars = ps_rectangle.create_subregions()

    img_width, img_height = page.get_image_resolution()

    return ars, img_width, img_height


def get_article_rectangle_dict(ar_list=None):
    """Convert the list of ArticleRectangle objects `ar_list` to a dictionary with the article ids as keys and a list
    of ArticleRectangle objects as values.

    :param ar_list: list of ArticleRectangle objects
    :type ar_list: list of ArticleRectangle
    :return: a dictionary with article ids as keys and a list of ArticleRectangle objects as values
    """
    d = dict()
    for ar in ar_list:
        if len(ar.get_articles()) > 1:
            print(f"Expected at most one article id, but got {len(ar.get_articles())}.")
        else:
            if len(ar.get_articles()) == 0:
                a_id = "no_bl"
            else:
                a_id = list(ar.get_articles())[0]
            try:
                d[a_id].append(ar)
            except KeyError:
                d[a_id] = [ar]

    return d


def convert_dict_to_list(d):
    return list(d.values())


def convert_rectangle_list_to_poly_list(rect_list):
    """Takes a list of Rectangle objects `rect_list` as input and converts them to a list of lists of four (x,y) pairs
    for every Rectangle in rect_list.

    :param rect_list: a list of Rectangle objects
    :type rect_list: list of Rectangle
    :return: list of lists of four (x,y) pairs each representing the corners of the rectangles
    """
    return [[(ar.x, ar.y), (ar.x + ar.width, ar.y), (ar.x + ar.width, ar.y + ar.height),
             (ar.x, ar.y + ar.height)] for ar in rect_list]


def plot_gt_data(img_path, article_rect_dict, img_width, img_height, kernel_size=4, savedir=None):
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
    plot.add_image(ax, img_path)

    poly_gt_img = Image.new('1', (img_width, img_height))

    for i, a_id in enumerate(article_rect_dict):
        # add facecolors="None" if rectangles should not be filled
        ar_poly_list = convert_rectangle_list_to_poly_list(article_rect_dict[a_id])
        if a_id == "no_bl":
            ar_poly_collection = PolyCollection(ar_poly_list, closed=True, edgecolors='k', facecolors='k')
        else:
            ar_poly_collection = PolyCollection(ar_poly_list, closed=True, edgecolors=COLORS[i], facecolors=COLORS[i])
        ar_poly_collection.set_alpha(0.5)
        ax.add_collection(ar_poly_collection)

        poly_gt_img = plot_polys_binary(ar_poly_list, poly_gt_img)

    poly_gt_img = apply_dilation(poly_gt_img, kernel_size)

    if savedir is None:
        poly_gt_img.show()
    else:
        poly_gt_img.save(savedir)


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


def apply_dilation(img, kernel_size=4):
    """Applies dilation with a kernel of size `kernel_size x kernel_size` (all entries filled with ones) to the image
    given by `img`and returns the dilated image with the same type as the input

    :param img: input image to perform dilation on
    :type img: Union[Image.Image, np.array]
    :param kernel_size: size of the kernel filter
    :return: the dilated image
    """
    if kernel_size == 0:
        return img
    # define kernel for dilation
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = np.ones((kernel_size, 1), np.uint8)
    # if image is a PIL Image, convert it to a numpy array, so it can be used in opencv
    if type(img) == Image.Image:
        # Convert to numpy array to use it in opencv (to apply dilation later)
        img = img.convert('RGB')
        img = np.array(img)

        return Image.fromarray(cv2.dilate(img, kernel, iterations=1))

    return cv2.dilate(img, kernel, iterations=1)


def apply_erosion(img, kernel_size=4):
    """Applies dilation with a kernel of size `kernel_size x kernel_size` (all entries filled with ones) to the image
    given by `img`and returns the dilated image with the same type as the input

    :param img: input image to perform dilation on
    :type img: Union[Image.Image, np.array]
    :param kernel_size: size of the kernel filter
    :return: the dilated image
    """
    if kernel_size == 0:
        return img
    # define kernel for dilation
    # kernel = np.ones((kernel_size, kernel_size), np.uint8)
    kernel = np.ones((kernel_size, 1), np.uint8)
    # if image is a PIL Image, convert it to a numpy array, so it can be used in opencv
    if type(img) == Image.Image:
        # Convert to numpy array to use it in opencv (to apply dilation later)
        img = img.convert('RGB')
        img = np.array(img)

        return Image.fromarray(cv2.erode(img, kernel, iterations=1))

    return cv2.erode(img, kernel, iterations=1)


if __name__ == '__main__':
    # # path_to_page = "/home/max/data/as/newseye_as_test_data/xml_files_gt/19000715_1-0001.xml"
    # # path_to_img = "/home/max/data/as/newseye_as_test_data/image_files/19000715_1-0001.jpg"
    # # path_to_img = "./test/resources/newseye_as_test_data/image_files/0033_nzz_18120804_0_0_a1_p1_1.tif"
    # # path_to_page = "./test/resources/newseye_as_test_data/xml_files_gt/0033_nzz_18120804_0_0_a1_p1_1.xml"
    #
    # path_to_img = "/home/max/data/as/NewsEye_ONB_Data/140878/ONB_krz_19330701/ONB_krz_19330701_001.jpg"
    # path_to_page = "/home/max/data/as/NewsEye_ONB_Data/140878/ONB_krz_19330701/page/ONB_krz_19330701_001.xml"
    # ars, img_width, img_height = get_article_rectangles(path_to_page)
    #
    # ars_dict = get_article_rectangle_dict(ars)
    #
    # plot_gt_data(path_to_img, ars_dict, img_width, img_height + 1, kernel_size=5,
    #              savedir="./test/resources/test_as_gt_function.jpg")
    #
    # # for ar in ars:
    # #     print(ar.a_ids, ar.x, ar.y, ar.width, ar.height)

    polygon_example = [(0, 0), (50, 0), (50, 20), (30, 20), (30, 40), (50, 40), (50, 60), (0, 60), (0, 0)]
    polygon_img = plot_polys_binary([polygon_example], img_width=100, img_height=100)

    polygon_img = apply_dilation(polygon_img, kernel_size=20)
    polygon_img = apply_erosion(polygon_img, kernel_size=20)
    polygon_img.show()
