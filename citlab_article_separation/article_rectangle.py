from citlab_python_util.geometry.rectangle import Rectangle
from citlab_python_util.geometry.util import check_intersection


class ArticleRectangle(Rectangle):

    def __init__(self, x=0, y=0, width=0, height=0, textlines=None, article_ids=None):
        super().__init__(x, y, width, height)

        self.textlines = textlines
        if article_ids is None and textlines is not None:
            self.a_ids = self.get_articles()
        else:
            self.a_ids = article_ids

    def get_articles(self):
        # traverse the baselines/textlines and check the article id
        article_set = set()

        for tl in self.textlines:
            article_set.add(tl.get_article_id())

        return article_set

    def contains_polygon(self, polygon, x, y, width, height):
        """ Checks if a polygon intersects with (or lies within) a (sub)rectangle given by the coordinates x,y
        (upper left point) and the width and height of the rectangle. """

        # iterate over the points of the polygon
        for i in range(polygon.n_points - 1):
            line_segment_bl = [polygon.x_points[i:i + 2], polygon.y_points[i:i + 2]]

            # The baseline segment lies outside the rectangle completely to the right/left/top/bottom
            if max(line_segment_bl[0]) <= x or min(line_segment_bl[0]) >= x + width or max(
                    line_segment_bl[1]) <= y or min(line_segment_bl[1]) >= y + height:
                continue

            # The baseline segment lies inside the rectangle
            if min(line_segment_bl[0]) >= x and max(line_segment_bl[0]) <= x + width and min(
                    line_segment_bl[1]) >= y and max(line_segment_bl[1]) <= y + height:
                return True

            # The baseline segment intersects with the rectangle or lies outside the rectangle but doesn't lie
            # completely right/left/top/bottom
            # First check intersection with the vertical line segments of the rectangle
            line_segment_rect_left = [[x, x], [y, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_left) is not None:
                return True
            line_segment_rect_right = [[x + width, x + width], [y, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_right) is not None:
                return True

            # Check other two sides
            line_segment_rect_top = [[x, x + width], [y, y]]
            if check_intersection(line_segment_bl, line_segment_rect_top) is not None:
                return True
            line_segment_rect_bottom = [[x, x + width], [y + height, y + height]]
            if check_intersection(line_segment_bl, line_segment_rect_bottom) is not None:
                return True

        return False

    def create_subregions(self, ar_list=None):

        # width1 equals width2 if width is even, else width2 = width1 + 1
        # same for height1 and height2
        if ar_list is None:
            ar_list = []
        width1 = self.width // 2
        width2 = self.width - width1
        height1 = self.height // 2
        height2 = self.height - height1

        #########################
        #           #           #
        #     I     #     II    #
        #           #           #
        #########################
        #           #           #
        #    III    #     IV    #
        #           #           #
        #########################

        # determine textlines for each subregion
        tl1 = []
        tl2 = []
        tl3 = []
        tl4 = []
        a_ids1 = set()
        a_ids2 = set()
        a_ids3 = set()
        a_ids4 = set()

        for tl in self.textlines:
            # get baseline
            bl = tl.baseline.to_polygon()
            contains_polygon = self.contains_polygon(bl, self.x, self.y, width1, height1)
            if contains_polygon:
                tl1 += [tl]
                a_ids1.add(tl.get_article_id())
                # continue
            contains_polygon = self.contains_polygon(bl, self.x + width1, self.y, width2, height1)
            if contains_polygon:
                tl2 += [tl]
                a_ids2.add(tl.get_article_id())
                # continue
            contains_polygon = self.contains_polygon(bl, self.x, self.y + height1, width1, height2)
            if contains_polygon:
                tl3 += [tl]
                a_ids3.add(tl.get_article_id())
                # continue
            contains_polygon = self.contains_polygon(bl, self.x + width1, self.y + height1, width2, height2)
            if contains_polygon:
                tl4 += [tl]
                a_ids4.add(tl.get_article_id())
                # continue

        a_rect1 = ArticleRectangle(self.x, self.y, width1, height1, tl1, a_ids1)
        a_rect2 = ArticleRectangle(self.x + width1, self.y, width2, height1, tl2, a_ids2)
        a_rect3 = ArticleRectangle(self.x, self.y + height1, width1, height2, tl3, a_ids3)
        a_rect4 = ArticleRectangle(self.x + width1, self.y + height1, width2, height2, tl4, a_ids4)

        # run create_subregions on Rectangles that contain more than one TextLine object
        for a_rect in [a_rect1, a_rect2, a_rect3, a_rect4]:
            if len(a_rect.a_ids) > 1:  # or a_rect.width > 200:
                a_rect.create_subregions(ar_list)
            else:
                ar_list.append(a_rect)

        return ar_list
