import numpy as np
from shapely import geometry

from citlab_article_separation.net_post_processing.region_to_page_writer import RegionToPageWriter
from citlab_python_util.parser.xml.page.page_constants import sTEXTREGION
from citlab_python_util.parser.xml.page.page_objects import TextRegion


class TextRegionToPageWriter(RegionToPageWriter):
    def __init__(self, path_to_page, path_to_image=None, fixed_height=None, scaling_factor=None, region_dict=None):
        super().__init__(path_to_page, path_to_image, fixed_height, scaling_factor)
        self.region_dict = region_dict

    def remove_text_regions_from_page(self):
        self.remove_region_by_type(sTEXTREGION)

    def overwrite_text_regions(self):
        print("START OVERWRITING OF TEXT LINES")
        text_lines = self.page_object.get_textlines(self.page_object.page_doc)
        self.remove_text_regions_from_page()
        text_regions = self.region_dict[sTEXTREGION]

        print("Creating text region objects and shapely Polygons")
        text_regions_obj, text_regions_sh = [], []
        for i in range(len(text_regions)):
            text_region_id = sTEXTREGION + "_" + str(i + 1)
            text_regions_obj.append(TextRegion(text_region_id, points=text_regions[i], text_lines=[]))
            text_regions_sh.append(geometry.Polygon(text_regions[i]).buffer(0))

        print("Iterate over the text lines")
        for text_line in text_lines:
            text_line_sh = geometry.Polygon(text_line.surr_p.points_list)
            text_region_idx = int(np.argmax([text_line_sh.intersection(text_region_sh).area
                                             for text_region_sh in text_regions_sh]))
            if text_region_idx is not None:
                text_regions_obj[text_region_idx].text_lines.append(text_line)

        for text_region_obj in text_regions_obj:
            self.page_object.add_region(text_region_obj)
