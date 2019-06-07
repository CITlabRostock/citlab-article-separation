from citlab_python_util.geometry.util import ortho_connect


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