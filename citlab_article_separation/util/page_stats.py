import argparse
import os
import pandas
from citlab_python_util.parser.xml.page.page import Page
import citlab_python_util.parser.xml.page.page_constants as page_constants
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")


def get_text_region_article_dict(path_to_pagexml=None, page=None):
    if (path_to_pagexml is None and page is None) or (path_to_pagexml is not None and page is not None):
        raise ValueError(f"Either path to pageXML or Page object is needed!")
    if path_to_pagexml:
        page = Page(path_to_pagexml)
    text_regions = page.get_text_regions()
    article_dict = dict()
    for text_region in text_regions:
        # assume that all text lines of the region have the same article ID
        a_id = text_region.text_lines[0].get_article_id()
        article_dict[text_region.id] = a_id
    return article_dict


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
    res["ID"] = os.path.splitext(os.path.basename(path_to_pagexml))[0]
    res["ImageWidth"] = width
    res["ImageHeight"] = height

    # get regions
    dict_of_regions = page_file.get_regions()
    if article_stats:
        article_dict = page_file.get_article_dict()
        logger.info(f"- Number of articles: {len(set(article_dict.keys()))}")
        res["Articles"] = len(set(article_dict.keys()))

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

    with open(args.pagexml_list, "r") as list_file:
        page_data = []
        for path in list_file:
            stats = get_page_stats(path.rstrip(), args.region_stats, args.text_line_stats, args.article_stats)
            page_data.append(list(stats.values()))
        column_keys = list(stats.keys())

    if args.csv_path:
        df = pandas.DataFrame(page_data, columns=column_keys)
        nan_columns = df.columns[df.isna().any()].tolist()  # check for columns containing NaN
        if nan_columns:
            nan_dict = dict(zip(nan_columns, ["int32"] * len(nan_columns)))
            # replace NaN with 0 and cast columns to int (is float after fillna)
            df = df.fillna(value=0).astype(nan_dict)
            logger.info(f"Replaced NaN values with 0 in columns {nan_columns}")
        df.to_csv(args.csv_path, index=False)
        logger.info(f"Wrote data to {args.csv_path}")
