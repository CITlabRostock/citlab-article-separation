import argparse
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
    # load the page xml file
    logger.info(f"Processing {path_to_pagexml}")
    page_file = Page(path_to_pagexml)
    width, height = page_file.get_image_resolution()
    logger.info(f"- Image resolution: width={width}, height={height}")

    # get regions
    dict_of_regions = page_file.get_regions()
    if region_stats:
        for key in dict_of_regions:
            regions = dict_of_regions[key]
            if text_line_stats and key == page_constants.sTEXTREGION:
                text_lines = []
                for text_region in regions:
                    text_lines.extend(text_region.text_lines)
                logger.info(f"- Number of {key}: {len(regions)}, number of text_lines: {len(text_lines)}")
            else:
                logger.info(f"- Number of {key}: {len(dict_of_regions[key])}")

    if article_stats:
        article_dict = page_file.get_article_dict()
        logger.info(f"- Number of articles: {len(set(article_dict.keys()))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pagexml_list', help="Input list with paths to pagexml files", required=True)
    parser.add_argument('--region_stats', type=bool, default=True, metavar="BOOL",
                        help="Get region stats or not.")
    parser.add_argument('--text_line_stats', type=bool, default=True, metavar="BOOL",
                        help="Get text_line stats or not.")
    parser.add_argument('--article_stats', type=bool, default=True, metavar="BOOL",
                        help="Get article stats or not.")
    args = parser.parse_args()

    with open(args.pagexml_list, "r") as list_file:
        for path in list_file:
            get_page_stats(path.rstrip())
