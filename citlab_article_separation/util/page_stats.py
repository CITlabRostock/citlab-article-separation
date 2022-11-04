import argparse
from pathlib import Path
import pandas
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_objects import REGIONS_DICT
import citlab_python_util.parser.xml.page.page_constants as page_constants
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")


def get_page_stats(path_to_pagexml, region_stats=True, text_line_stats=True, article_stats=True):
    """ Extraction of the information contained by a given Page xml file.

    :param path_to_pagexml: file path of the Page xml
    :return: list of polygons, list of article ids, image resolution
    """
    res = dict()

    # load the page xml file
    logger.info(f"Processing {path_to_pagexml}")
    page_file = Page(path_to_pagexml)
    width, height = page_file.get_image_resolution()
    logger.info(f"- Image resolution: width={width}, height={height}")
    res["ID"] = Path(path_to_pagexml).stem
    res["ImageWidth"] = width
    res["ImageHeight"] = height

    # get regions
    dict_of_regions = page_file.get_regions()
    if article_stats:
        article_region_dict, _ = page_file.get_article_region_dicts()
        num_articles = len(article_region_dict)
        logger.info(f"- Number of articles: {num_articles}")
        res["Articles"] = num_articles

    if region_stats:
        for key in dict_of_regions:
            regions = dict_of_regions[key]
            if text_line_stats and key == page_constants.sTEXTREGION:
                text_lines = []
                for text_region in regions:
                    text_lines.extend(text_region.text_lines)
                logger.info(f"- Number of {key}: {len(regions)}, number of text_lines: {len(text_lines)}")
                res[key] = len(regions)
                res["TextLine"] = len(text_lines)
            else:
                logger.info(f"- Number of {key}: {len(dict_of_regions[key])}")
                res[key] = len(dict_of_regions[key])
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pagexml_list', help="Input list with paths to pagexml files", required=True)
    parser.add_argument('--csv_path', default="", help="Optionally save data in csv given by this path")
    parser.add_argument('--region_stats', type=bool, default=True, metavar="BOOL",
                        help="Get region stats or not.")
    parser.add_argument('--text_line_stats', type=bool, default=True, metavar="BOOL",
                        help="Get text_line stats or not.")
    parser.add_argument('--article_stats', type=bool, default=True, metavar="BOOL",
                        help="Get article stats or not.")
    args = parser.parse_args()

    page_paths = [line.rstrip() for line in open(args.pagexml_list, "r")]

    # get region indices
    column_keys = ["ID", "ImageWidth", "ImageHeight", "Articles", "TextLine"]
    column_keys.extend(REGIONS_DICT.keys())

    # get page stats
    page_data = []
    for path in page_paths:
        page_dict = dict.fromkeys(tuple(column_keys))
        stats = get_page_stats(path, args.region_stats, args.text_line_stats, args.article_stats)
        page_dict.update(stats)
        page_data.append(list(page_dict.values()))

    # build dataframe and save csv
    if args.csv_path:
        # df = pandas.DataFrame.from_dict(dict(zip(region_keys, page_data)), orient="index", columns=region_keys)
        df = pandas.DataFrame(page_data, columns=column_keys)
        # remove columns without any values
        all_nan_columns = df.columns[df.isna().all()].tolist()  # check for columns containing all NaN's
        if all_nan_columns:
            df.drop(columns=all_nan_columns, inplace=True)
            logger.info(f"Removed columns that only contained NaN's: {all_nan_columns}")
        # replace remaining NaN's
        any_nan_columns = df.columns[df.isna().any()].tolist()  # check for columns containing any NaN
        if any_nan_columns:
            nan_dict = dict(zip(any_nan_columns, ["int32"] * len(any_nan_columns)))
            # replace NaN with 0 and cast columns to int (is float after fillna)
            df = df.fillna(value=0).astype(nan_dict)
            logger.info(f"Replaced NaN values with 0 in columns {any_nan_columns}")
        df.to_csv(args.csv_path, index=False)
        logger.info(f"Wrote data to {args.csv_path}")
        df_stats = df.describe().round(decimals=2)
        csv_path = Path(args.csv_path)
        csv_stats_path = csv_path.parent / f"{csv_path.stem}_describe.csv"
        df_stats.to_csv(csv_stats_path, index=True)
        logger.info(f"Wrote statistics data to {csv_stats_path}")
