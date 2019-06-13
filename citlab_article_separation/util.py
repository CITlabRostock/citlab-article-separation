from citlab_python_util.geometry.util import ortho_connect
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_objects import Points

from citlab_article_separation.article_rectangle import ArticleRectangle


def get_article_surrounding_polygons(ar_dict):
    """
    Create surrounding polygons over a sets of rectangles, belonging to different article_ids.
    :param ar_dict: dict (keys = article_id, values = corresponding rectangles)
    :return: dict (keys = article_id, values = corresponding surrounding polygons)
    """
    asp_dict = {}
    for id in ar_dict:
        sp = ortho_connect(ar_dict[id])
        asp_dict[id] = sp
    return asp_dict


def get_article_rectangles(page):
    """Given the PageXml file `page` return the corresponding article subregions as a list of ArticleRectangle objects.
     Also returns the width and height of the image (NOT of the PrintSpace).

    :param page: Either the path to the PageXml file or a Page object.
    :type page: Union[str, Page]
    :return: the article subregion list, the height and the width of the image
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

    return ars, img_height, img_width
