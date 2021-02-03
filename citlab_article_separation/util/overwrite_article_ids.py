import argparse
import os
import re
import numpy as np

from citlab_python_util.parser.xml.page.page import Page


def overwrite_article_ids(page_list, gt_list):
    print("Load pagexml files in {} and overwrite textline article_ids from {}".format(page_list, gt_list))
    page_files = open(page_list, "r")
    gt_files = open(gt_list, "r")

    page_list = page_files.readlines()
    gt_list = gt_files.readlines()

    assert len(page_list) == len(gt_list), \
        "Page list and GT list must have the same number of elements: {} != {}".format(len(page_list), len(gt_list))

    # sort path lists by file name
    page_list = sorted(page_list, key=os.path.basename)
    gt_list = sorted(gt_list, key=os.path.basename)

    all_update_counter = 0
    file_counter = 0
    for page_path, gt_path in zip(page_list, gt_list):
        if os.path.basename(page_path.rstrip()) != os.path.basename(gt_path.rstrip()):
            raise ValueError("Page and GT file names do not match")

        # load the page xml files
        page_path = os.path.abspath(page_path.rstrip())
        gt_path = os.path.abspath(gt_path.rstrip())
        try:
            page_file = Page(page_path)
        except Exception as ex:
            print("Defined PAGEXML \"{}\" can not be loaded!\n{}".format(page_path, ex))
            continue
        try:
            gt_file = Page(gt_path)
        except Exception as ex:
            print("Defined PAGEXML \"{}\" can not be loaded!\n{}".format(gt_path, ex))
            continue

        # build GT dict with textline ids as keys and their article_ids as values
        gt_textlines = gt_file.get_textlines()
        gt_article_dict = dict()
        for tl in gt_textlines:
            gt_article_dict[tl.id] = tl.get_article_id()

        # go over page textlines and overwrite article_id with GT if necessary
        update_counter = 0
        page_textlines = page_file.get_textlines()
        for tl in page_textlines:
            if tl.get_article_id() != gt_article_dict[tl.id]:
                # print("Update textline ({}): article_id old({}) -> new({})".format(tl.id, tl.get_article_id(), gt_article_dict[tl.id]))
                tl.set_article_id(gt_article_dict[tl.id])
                update_counter += 1
        page_file.set_textline_attr(page_textlines)
        print("Updated {}/{} textline article_ids in {}".format(update_counter, len(page_textlines), ".../" + os.path.basename(page_path)))
        all_update_counter += update_counter

        assert all([tl.get_article_id() == gt_article_dict[tl.id] for tl in page_file.get_textlines()]), \
            "Overwritten article_ids do not match GT article_ids. Something went wrong."

        # write pagexml
        if update_counter > 0:
            page_file.write_page_xml(page_path)
            print("Wrote updated pagexml to {}".format(page_path))
            file_counter += 1

    print("Done!\nUpdated {}/{} files and overall {} textline article_ids".format(file_counter, len(page_list), all_update_counter))
    page_files.close()
    gt_files.close()


def overwrite_article_ids_by_region(page_list, gt_list):
    print("Load pagexml files in {} and overwrite textline article_ids from {}".format(page_list, gt_list))
    page_files = open(page_list, "r")
    gt_files = open(gt_list, "r")

    page_list = page_files.readlines()
    gt_list = gt_files.readlines()

    assert len(page_list) == len(gt_list), \
        "Page list and GT list must have the same number of elements: {} != {}".format(len(page_list), len(gt_list))

    # sort path lists by file name
    page_list = sorted(page_list, key=os.path.basename)
    gt_list = sorted(gt_list, key=os.path.basename)
    # for p, g in zip(page_list, gt_list):
    #     print(f"{p} ------ {g}")

    num_empty_regions = 0
    num_degenerate = 0
    num_text_regions = 0
    num_removed_regions = 0
    for page_path, gt_path in zip(page_list, gt_list):
        # load the page xml files
        page_path = os.path.abspath(page_path.rstrip())
        gt_path = os.path.abspath(gt_path.rstrip())

        # if "576455_0004_23676310" not in page_path:
        #     continue

        try:
            page_file = Page(page_path)
        except Exception as ex:
            print("Defined PAGEXML \"{}\" can not be loaded!\n{}".format(page_path, ex))
            continue
        try:
            gt_file = Page(gt_path)
        except Exception as ex:
            print("Defined PAGEXML \"{}\" can not be loaded!\n{}".format(gt_path, ex))
            continue

        # Make sure files match
        page_img = page_file.metadata.TranskribusMeta.imageId
        gt_img = gt_file.metadata.TranskribusMeta.imageId
        if page_img != gt_img:
            raise ValueError(f"Page and GT file image reference mismatch (Page: {page_img} - GT: {gt_img})")

        print(f"Updating {page_path}")
        # build GT dict with textline ids as keys and their article_ids as values
        gt_text_regions = gt_file.get_text_regions()
        gt_article_dict = dict()
        print(f"GT: {gt_path}")
        for text_region in gt_text_regions:
            if len(text_region.text_lines) == 0:
                # print(f"{gt_path} - {text_region.id} - contains no text_lines. Skipping.")
                num_empty_regions += 1
                tr_points = np.array(text_region.points.points_list, dtype=np.int32)
                # bounding box of text region
                min_x = np.min(tr_points[:, 0])
                min_y = np.min(tr_points[:, 1])
                max_x = np.max(tr_points[:, 0])
                max_y = np.max(tr_points[:, 1])
                size_x = float(max_x) - float(min_x)
                size_y = float(max_y) - float(min_y)
                if size_x + size_y < 10:
                    num_degenerate += 1
                continue
            text_region_article_ids = []
            for text_line in text_region.text_lines:
                if text_line.get_article_id() is not None:
                    text_region_article_ids.append(text_line.get_article_id())
            if not text_region_article_ids:
                # print(f"{gt_path} - {text_region.id} - contains no article_IDs. Skipping.")
                continue
            values, counts = np.unique(text_region_article_ids, return_counts=True)
            index = np.argmax(counts)
            # if len(values) > 1:
            #     print(f"{gt_path} - {text_region.id} - contains multiple article IDs ({text_region_article_ids})")
            gt_article_dict[text_region.id] = values[index]

        # go over page text regions and overwrite respective text_line article_ids with GT
        page_text_regions = page_file.get_text_regions()
        num_text_regions += len(page_text_regions)
        updated_text_regions = []
        for text_region in page_text_regions:
            if len(text_region.text_lines) == 0:
                # print(f"{page_path} - {text_region.id} - contains no text_lines. Skipping.")
                num_removed_regions += 1
                continue
            try:
                article_id = gt_article_dict[text_region.id]
                # print(text_region.id, article_id)
            except KeyError:
                # print(f"{page_path} - {text_region.id} - found no matching text_region in GT.")
                num_removed_regions += 1
                continue
            for text_line in text_region.text_lines:
                text_line.set_article_id(article_id)
            updated_text_regions.append(text_region)

        for text_region in updated_text_regions:
            assert all([text_line.get_article_id() == gt_article_dict[text_region.id]
                        for text_line in text_region.text_lines])
        page_file.set_text_regions(updated_text_regions, overwrite=True)

        # write pagexml
        page_file.write_page_xml(page_path)
        print(f"Wrote updated pagexml to {page_path}")

    print(f"GT pages contained {num_empty_regions} TextRegions without baselines, "
          f"from which {num_degenerate} are degenerate (size < 10).")
    print(f"From original {num_text_regions} TextRegions, {num_removed_regions} were removed "
          f"due to missing baselines or article_ids.")
    page_files.close()
    gt_files.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_list', help="Input list with paths to pagexml files", required=True)
    parser.add_argument('--gt_list', help="GT list with paths to corresponding pagexml files", required=True)
    args = parser.parse_args()

    # overwrite_article_ids(args.in_list, args.gt_list)
    overwrite_article_ids_by_region(args.in_list, args.gt_list)
