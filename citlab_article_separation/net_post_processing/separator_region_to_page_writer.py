from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
from shapely import geometry

from citlab_article_separation.net_post_processing.region_to_page_writer import RegionToPageWriter
from citlab_python_util.parser.xml.page.page_constants import sSEPARATORREGION, sTEXTREGION
from citlab_python_util.parser.xml.page.page_objects import SeparatorRegion
from citlab_python_util.parser.xml.page.plot import plot_pagexml


class SeparatorRegionToPageWriter(RegionToPageWriter):
    def __init__(self, path_to_page, region_dict, image_path=None, fixed_height=None, scaling_factor=None):
        super().__init__(path_to_page, region_dict, image_path, fixed_height, scaling_factor)

    def merge_regions(self):
        def _split_shapely_polygon(region_to_split_sh, region_compare_sh):
            difference = region_to_split_sh.difference(region_compare_sh)
            if type(difference) == geometry.MultiPolygon or type(difference) == geometry.MultiLineString:
                new_region_polys_sh = list(difference)
            else:
                new_region_polys_sh = [difference]

            return new_region_polys_sh

        def _create_page_objects(region_to_split, new_region_polys):
            new_region_objects = [deepcopy(region_to_split) for _ in range(len(new_region_polys))]

            for j, (new_region_poly, new_region_object) in enumerate(
                    zip(new_region_polys, new_region_objects)):
                new_region_object.set_points(new_region_poly)
                if len(new_region_polys) > 1:
                    new_region_object.id = region_to_split.id + "_" + str(j + 1)

            return new_region_objects

        def _delete_region_from_page(region_id):
            region_to_delete = self.page_object.get_child_by_id(self.page_object.page_doc, region_id)
            if len(region_to_delete) == 0:
                return
            region_to_delete = region_to_delete[0]
            self.page_object.remove_page_xml_node(region_to_delete)

        def _add_regions_to_page(region_object_list):
            for region_object in region_object_list:
                self.page_object.add_region(region_object)

        def _get_parent_region(child_split_sh, parent_splits_sh):
            for j, parent_split_sh in enumerate(parent_splits_sh):
                if child_split_sh.intersects(parent_split_sh):
                    return j, parent_split_sh
            return None, None

        def _split_regions(region_dict, sep_poly):
            """
            Given a SeparatorRegion, split regions in region_dict if possible/necessary. Returns False if one of the
            regions in `region_dict` contains the SeparatorRegion. Then don't write it to the PAGE file.
            This function assumes, that the text lines lie completely within the text regions and the baselines lie
            completely within the text lines.
            :param region_dict:
            :param sep_poly:
            :return:
            """
            sep_poly_sh = geometry.Polygon(sep_poly).buffer(0)
            if type(sep_poly_sh) == geometry.MultiPolygon:
                sep_poly_sh = sep_poly_sh[np.argmax([poly.area for poly in list(sep_poly_sh)])]
                # sep_poly_sh = sep_poly_sh[max(range(len(list(sep_poly_sh))), key=lambda i: list(sep_poly_sh)[i].area)]

            for region_type, region_list in region_dict.items():
                updated_region_list = deepcopy(region_list)
                all_new_region_objects = []
                for i, region in enumerate(region_list):
                    region_polygon_sh = geometry.Polygon(region.points.points_list)
                    if region_polygon_sh.intersects(sep_poly_sh):
                        if region_polygon_sh.contains(sep_poly_sh):
                            # don't need to check the other regions, provided that we don't have overlapping regions
                            return False

                        new_region_polys_sh = _split_shapely_polygon(region_polygon_sh, sep_poly_sh)
                        new_region_polys = [list(poly.exterior.coords) for poly in new_region_polys_sh]
                        new_region_objects = _create_page_objects(region, new_region_polys)

                        # if the region is a TextRegion we also need to take care of the baselines and text lines
                        if region_type == sTEXTREGION:
                            for new_region_object in new_region_objects:
                                new_region_object.text_lines = []

                            text_lines = region.text_lines
                            for text_line in text_lines:
                                text_line_sh = geometry.Polygon(text_line.surr_p.points_list).buffer(0)
                                if text_line_sh.intersects(sep_poly_sh):
                                    text_line_splits_sh = _split_shapely_polygon(text_line_sh, sep_poly_sh)
                                    text_line_splits = [list(poly.exterior.coords) for poly in text_line_splits_sh]

                                    new_text_line_objects = _create_page_objects(text_line, text_line_splits)
                                    for new_text_line_object in new_text_line_objects:
                                        new_text_line_object.set_baseline(None)
                                        new_text_line_object.words = []

                                    # word_idx = np.argmax(
                                    #     [geometry.Polygon(word.surr_p.points_list).buffer(0).distance(sep_poly_sh)
                                    #      for word in text_line.words])

                                    for word in text_line.words:
                                        word_polygon_sh = geometry.Polygon(word.surr_p.points_list).buffer(0)
                                        matching_textline_idx = np.argmax([word_polygon_sh.intersection(text_line_split_sh)
                                        for text_line_split_sh in text_line_splits_sh])
                                        corr_textline = new_text_line_objects[matching_textline_idx]
                                        corr_textline.words.append(word)

                                    if len(text_line.words) > 0:
                                        for new_text_line_object in new_text_line_objects:
                                            new_text_line_object.text = " ".join([word.text for word in text_line.words])

                                    baseline_sh = geometry.LineString(
                                        text_line.baseline.points_list) if text_line.baseline is not None else None
                                    if baseline_sh is not None and baseline_sh.intersects(sep_poly_sh):
                                        baseline_splits = _split_shapely_polygon(baseline_sh, sep_poly_sh)

                                        # baseline split -> text line split
                                        for baseline_split in baseline_splits:
                                            idx, parent_text_line = _get_parent_region(baseline_split,
                                                                                       text_line_splits_sh)
                                            if idx is None:
                                                continue
                                            new_text_line_objects[idx].set_baseline(list(baseline_split.coords))

                                else:
                                    text_line_splits_sh = [text_line_sh]
                                    new_text_line_objects = [text_line]

                                # text line split -> region split
                                for text_line_split, new_text_line_object in zip(text_line_splits_sh,
                                                                                 new_text_line_objects):
                                    idx, parent_region = _get_parent_region(text_line_split, new_region_polys_sh)
                                    if idx is None:
                                        continue
                                    new_region_objects[idx].text_lines.append(new_text_line_object)

                        _delete_region_from_page(region.id)
                        offset = len(region_list) - len(updated_region_list)
                        updated_region_list.pop(i - offset)

                        # updated_region_list[i:i + 1] = new_region_objects

                        all_new_region_objects.extend(new_region_objects)
                        _add_regions_to_page(new_region_objects)
                updated_region_list.extend(all_new_region_objects)
                region_dict[region_type] = updated_region_list

                # _add_regions_to_page(all_new_region_objects)

                return True

        page_regions = self.page_object.get_regions()

        # For now we are only interested in the SeparatorRegion information
        separator_polygons = self.region_dict[sSEPARATORREGION]
        for separator_polygon in separator_polygons:
            use_separator = _split_regions(page_regions, separator_polygon)
            if use_separator is not True:
                continue

            separator_id = self.page_object.get_unique_id(sSEPARATORREGION)
            separator_region = SeparatorRegion(separator_id, points=separator_polygon)
            self.page_object.add_region(separator_region)

    def save_page_xml(self, save_path):
        self.page_object.write_page_xml(save_path)


if __name__ == '__main__':
    page_path = "/home/max/Downloads/transkribus_downloads/splitting_regions_example/splitting_regions_example/page/" \
                "DerGemeindebote-p02.xml"
    image_path = "/home/max/Downloads/transkribus_downloads/splitting_regions_example/splitting_regions_example/" \
                 "DerGemeindebote-p02.png"
    region_dict = {sSEPARATORREGION: [[(1745, 193), (2142, 193), (2142, 2784), (1745, 2784)]]}
    region_writer = SeparatorRegionToPageWriter(page_path, region_dict)

    old_regions = region_writer.page_object.get_regions()
    for region_type, region_list in old_regions.items():
        for region in region_list:
            print(region.id)
            print(region.points.points_list)
            if region_type == sTEXTREGION:
                print("+" * 10)
                print([text_line.id for text_line in region.text_lines])
                print([text_line.surr_p.points_list for text_line in region.text_lines])
                print([text_line.baseline.points_list for text_line in region.text_lines])
                print("+" * 10)

    print()
    print(region_dict)
    print()

    # page_old = region_writer.page_object
    # plot_pagexml(page_old, image_path, plot_article=False, plot_legend=False, fill_regions=True,
    #              use_page_image_resolution=True)
    # plt.show()

    region_writer.merge_regions()
    new_regions = region_writer.page_object.get_regions()
    for region_type, region_list in new_regions.items():
        for region in region_list:
            print(region.id)
            print(region.points.points_list)
            if region_type == sTEXTREGION:
                print("+" * 10)
                print([text_line.id for text_line in region.text_lines])
                print([text_line.surr_p.points_list for text_line in region.text_lines])
                print([text_line.baseline.points_list for text_line in region.text_lines])
                print("+" * 10)

    # region_writer.save_page_xml(save_path="/home/max/page_example.xml")

    region_writer.page_object.update_textlines()
    page_new = region_writer.page_object
    plot_pagexml(page_new, image_path, plot_article=False, plot_legend=False, fill_regions=True,
                 use_page_image_resolution=True)
    plt.show()
