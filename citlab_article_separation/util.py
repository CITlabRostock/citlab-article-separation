from citlab_python_util.geometry.util import ortho_connect, smooth_surrounding_polygon


def get_article_surrounding_polygons(ar_dict):
    """
    Create surrounding polygons over sets of rectangles, belonging to different article_ids.

    :param ar_dict: dict (keys = article_id, values = corresponding rectangles)
    :return: dict (keys = article_id, values = corresponding surrounding polygons)
    """
    asp_dict = {}
    for id in ar_dict:
        sp = ortho_connect(ar_dict[id])
        asp_dict[id] = sp
    return asp_dict


def smooth_article_surrounding_polygons(asp_dict, poly_norm_dist=10, or_dims=(400, 800, 600, 400)):
    """
    Create smoothed polygons over "crooked" polygons, belonging to different article_ids.

    1.) The polygon gets normalized, where the resulting vertices are at most `poly_norm_dist` pixels apart.

    2.) For each vertex of the original polygon an orientation is determined:

    2.1) Four rectangles (North, East, South, West) are generated, with the dimensions given by `or_dims`
    (width_vertical, height_vertical, width_horizontal, height_horizontal), i.e. North and South rectangles
    have dimensions width_v x height_v, whereas East and West rectangles have dimensions width_h x height_h.

    2.2) Each rectangle counts the number of contained points from the normalized polygon

    2.3) The top two rectangle counts determine the orientation of the vertex: vertical, horizontal or one
    of the four possible corner types.

    3.) Vertices with a differing orientation to its agreeing neighbours are assumed to be mislabeled and
    get its orientation converted to its neighbours.

    4.) Corner clusters of the same type need to be shrunken down to one corner, with the rest being converted
    to verticals. (TODO or horizontals)

    5.) Clusters between corners (corner-V/H-...-V/H-corner) get smoothed if they contain at least five points,
    by taking the average over the y-coordinates for horizontal edges and the average over the x-coordinates for
    vertical edges.

    :param asp_dict: dict (keys = article_id, values = list of "crooked" polygons)
    :param poly_norm_dist: int, distance between pixels in normalized polygon
    :param or_dims: tuple (width_v, height_v, width_h, height_h), the dimensions of the orientation rectangles
    :return: dict (keys = article_id, values = smoothed polygons)
    """
    asp_dict_smoothed = {}
    for id in asp_dict:
        sp_smooth = smooth_surrounding_polygon(asp_dict[id], poly_norm_dist, or_dims)
        asp_dict_smoothed[id] = sp_smooth
    return asp_dict_smoothed
