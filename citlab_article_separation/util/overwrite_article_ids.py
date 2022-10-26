import argparse
import os
from pathlib import Path
import numpy as np
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")


def find_paths(root=".", ending="xml", exclude=None):
    results = []
    for path in Path(root).rglob(f'*.{ending}'):
        results.append(str(path))
    if exclude:
        for ex in exclude.split(","):
            results = [res for res in results if ex not in res]
    return results


def overwrite_article_ids(page_paths, gt_paths, overwrite=False):
    page_paths = list(sorted(page_paths))
    gt_paths = list(sorted(gt_paths))
    gt_dict = dict([(str(Path(gt_path).stem), gt_path) for gt_path in gt_paths])

    all_update_counter = 0
    file_counter = 0
    for page_path in page_paths:
        # load the page xml files
        in_page = Page(page_path)
        page_key = str(Path(page_path).stem)  # file name
        gt_path = gt_dict[page_key]
        try:
            gt_page = Page(gt_path)
        except KeyError:
            logger.error(f"Missing matching GT file for IN file {page_path}. Skipping.")
            continue

        logger.info(f"Updating {page_path}")
        logger.info(f"GT: {gt_path}")

        # Get GT textline - article_id dict
        gt_article_dict = gt_page.get_article_textline_dict(inverse=True)

        # go over page textlines and overwrite article_id with GT if necessary
        update_counter = 0
        page_textlines = in_page.get_textlines()
        for tl in page_textlines:
            tl_id = tl.get_article_id()
            try:
                gt_tl_id = gt_article_dict[tl.id]
            except KeyError:
                logger.debug(f"\tTextline {tl.id} not found in GT file. Keeping article_id {tl.get_article_id()}.")
                continue
            if tl_id != gt_tl_id:
                logger.debug(f"\tUpdate textline ({tl.id}): article_id old({tl_id}) -> new({gt_tl_id})")
                tl.set_article_id(gt_tl_id)
                update_counter += 1

        in_page.set_textline_attr(page_textlines)
        logger.info(f"\tUpdated {update_counter}/{len(page_textlines)} textline article_ids in {page_path}")
        all_update_counter += update_counter

        # write pagexml
        if update_counter > 0:
            if overwrite:
                in_page.write_page_xml(page_path)
            else:
                page_path = Path(page_path).parent / (Path(page_path).stem + "_overwrite_articles.xml")
                in_page.write_page_xml(page_path)
            logger.info(f"\tWrote overwritten article IDs to {page_path}")
            file_counter += 1

    logger.info(f"Updated {file_counter}/{len(page_paths)} files and overall {all_update_counter} textline article_ids")


# TODO: update for page_paths
def overwrite_article_ids_by_region(page_paths, gt_paths):
    logger.info("Load pagexml files in {} and overwrite textline article_ids from {}".format(page_list, gt_list))
    page_files = open(page_list, "r")
    gt_files = open(gt_list, "r")

    page_list = page_files.readlines()
    gt_list = gt_files.readlines()

    assert len(page_list) == len(gt_list), \
        "Page list and GT list must have the same number of elements: {} != {}".format(len(page_list), len(gt_list))

    # sort path lists by file name
    page_list = sorted(page_list, key=os.path.basename)
    gt_list = sorted(gt_list, key=os.path.basename)

    num_empty_regions = 0
    num_degenerate = 0
    num_text_regions = 0
    num_removed_regions = 0
    for page_path, gt_path in zip(page_list, gt_list):
        # load the page xml files
        page_path = os.path.abspath(page_path.rstrip())
        gt_path = os.path.abspath(gt_path.rstrip())
        page_file = Page(page_path)
        gt_file = Page(gt_path)

        # Make sure files match
        page_img = page_file.metadata.TranskribusMeta.imageId
        gt_img = gt_file.metadata.TranskribusMeta.imageId
        # assert os.path.basename(page_path) == os.path.basename(gt_path)
        assert page_img == gt_img, f"Page and GT file image reference mismatch " \
                                   f"(Page: {page_img} - GT: {gt_img})\n{page_path}\n{gt_path}"

        logger.info(f"Updating {page_path}")
        logger.info(f"GT: {gt_path}")
        # build GT dict with textline ids as keys and their article_ids as values
        gt_text_regions = gt_file.get_text_regions()
        gt_article_dict = dict()
        for text_region in gt_text_regions:
            if len(text_region.text_lines) == 0:
                logger.warning(f"{gt_path} - {text_region.id} - contains no text_lines. Skipping.")
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
                logger.warning(f"{gt_path} - {text_region.id} - contains no article_IDs. Skipping.")
                continue
            values, counts = np.unique(text_region_article_ids, return_counts=True)
            index = np.argmax(counts)
            if len(values) > 1:
                logger.warning(f"{gt_path} - {text_region.id} - contains multiple article IDs "
                               f"({set(text_region_article_ids)}). Choosing maximum occurence ({values[index]}).")
            gt_article_dict[text_region.id] = values[index]

        # go over page text regions and overwrite respective text_line article_ids with GT
        page_text_regions = page_file.get_text_regions()
        num_text_regions += len(page_text_regions)
        updated_text_regions = []
        for text_region in page_text_regions:
            if len(text_region.text_lines) == 0:
                logger.warning(f"{page_path} - {text_region.id} - contains no text_lines. Removing.")
                num_removed_regions += 1
                continue
            try:
                article_id = gt_article_dict[text_region.id]
                logger.debug(text_region.id, article_id)
            except KeyError:
                logger.warning(f"{page_path} - {text_region.id} - found no matching text_region in GT. Removing.")
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
        logger.info(f"Wrote updated pagexml to {page_path}")

    logger.info(f"GT pages contained {num_empty_regions} TextRegions without baselines, "
                f"from which {num_degenerate} are degenerate (size < 10).")
    logger.info(f"From original {num_text_regions} TextRegions, {num_removed_regions} were removed "
                f"due to missing baselines or article_ids.")
    page_files.close()
    gt_files.close()


# TODO: update for page_paths
def clean_regions(page_paths):
    logger.info("Load pagexml files in {} and clean up regions".format(page_list))
    page_files = open(page_list, "r")
    page_list = page_files.readlines()

    num_text_regions = 0
    num_removed_textlines = 0
    num_removed_articles = 0
    for page_path in page_list:
        # load the page xml files
        page_path = os.path.abspath(page_path.rstrip())
        page_file = Page(page_path)
        logger.info(f"Updating {page_path}")

        # go over page text regions and check for missing textlines or article_ids
        page_text_regions = page_file.get_text_regions()
        num_text_regions += len(page_text_regions)
        updated_text_regions = []

        for text_region in page_text_regions:
            if len(text_region.text_lines) == 0:
                logger.warning(f"{page_path} - {text_region.id} - contains no text_lines. Removing.")
                num_removed_textlines += 1
                continue
            text_region_article_ids = []
            for text_line in text_region.text_lines:
                if text_line.get_article_id() is not None:
                    text_region_article_ids.append(text_line.get_article_id())
            if not text_region_article_ids:
                logger.warning(f"{page_path} - {text_region.id} - contains no article_IDs. Removing.")
                num_removed_articles += 1
                continue
            updated_text_regions.append(text_region)

        page_file.set_text_regions(updated_text_regions, overwrite=True)

        # write pagexml
        page_file.write_page_xml(page_path)
        logger.info(f"Wrote updated pagexml to {page_path}")

    logger.info(f"From original {num_text_regions} TextRegions, {num_removed_textlines + num_removed_articles} were "
                f"removed due to missing textlines ({num_removed_textlines}) or article_ids ({num_removed_articles}).")
    page_files.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--xml_list', help="Input list with paths to pagexml files (exclusive with --xml_dir)")
    parser.add_argument('--xml_dir', help="Input directory with pagexml files (exclusive with --xml_list)")
    parser.add_argument('--gt_list', help="GT list with paths to pagexml files (exclusive with --gt_dir)")
    parser.add_argument('--gt_dir', help="GT directory with pagexml files (exclusive with --gt_list)")
    parser.add_argument('--clean_regions', dest='clean_regions', default=False, action='store_true',
                        help="Clean regions with missing textlines or article_ids (default: False)")
    parser.add_argument('--overwrite_by_region', dest='overwrite_by_region', default=False, action='store_true',
                        help="Overwrite article IDs by region or by lines if clean_regions is False (default: False)")
    parser.add_argument('--overwrite', dest='overwrite', default=False, action='store_true',
                        help="Whether to overwrite the pageXML files or save new ones.")

    args = parser.parse_args()

    # INPUT variants
    if args.xml_dir and args.xml_list:
        logger.error(f"Only one XML input variant can be chosen at a time (either --xml_dir or --xml_list)!")
        exit(1)
    if not args.xml_dir and not args.xml_list:
        logger.error(f"Either --xml_dir or --xml_list is needed!")
        exit(1)
    if args.xml_dir:
        xml_paths = find_paths(root=args.xml_dir, exclude="_overwrite_articles")
        logger.info(f"Using XML directory '{set([str(Path(path).parent) for path in xml_paths])}'")
    else:  # args.xml_list
        xml_paths = [path.rstrip() for path in open(args.xml_list, "r")]
        logger.info(f"Using XML list '{args.xml_list}'")

    # GT variants
    if args.gt_dir and args.gt_list:
        logger.error(f"Only one XML GT variant can be chosen at a time (either --gt_dir or --gt_list)!")
        exit(1)
    if not args.gt_dir and not args.gt_list:
        logger.error(f"Either --gt_dir or --gt_list is needed!")
        exit(1)
    if args.gt_dir:
        gt_paths = find_paths(root=args.gt_dir, exclude="_overwrite_articles")
        logger.info(f"Using GT directory '{set([str(Path(path).parent) for path in gt_paths])}'")
    else:  # args.xml_list
        gt_paths = [path.rstrip() for path in open(args.gt_list, "r")]
        logger.info(f"Using GT list '{args.gt_list}'")

    if args.clean_regions:
        if args.overwrite_by_region:
            logger.warning(f"Both --clean_regions and --overwrite_by_region are True, we only clean regions for now.")
        else:
            logger.info(f"Cleaning regions...")
        clean_regions(xml_paths, args.overwrite)
    elif args.overwrite_by_region:
        logger.info(f"Overwriting article IDs by region...")
        overwrite_article_ids_by_region(xml_paths, gt_paths, args.overwrite)
    else:
        logger.info(f"Overwriting article IDs by textline...")
        overwrite_article_ids(xml_paths, gt_paths, args.overwrite)
