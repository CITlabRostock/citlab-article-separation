import json
import os
import argparse
import numpy as np
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.logging.custom_logging import setup_custom_logger

logger = setup_custom_logger(__name__, level="info")


def generate_finetuning_json(page_paths, json_path):
    xml_files = [line.rstrip('\n') for line in open(page_paths, "r")]
    json_dict = {"page": []}

    region_skips = 0
    for xml_file in xml_files:
        # load the page xml files
        page_file = Page(xml_file)
        page_name = os.path.splitext(os.path.basename(xml_file))[0]
        logger.info(f"Processing {xml_file}")

        article_id_text_region_id_dict, _ = page_file.get_article_region_dicts()

        article_list_of_dicts = []
        for article_id in article_id_text_region_id_dict:
            text_region_list_of_dicts = []

            for text_region_id in article_id_text_region_id_dict[article_id]:
                text_region_txt = ""
                text_region = page_file.get_region_by_id(text_region_id)
                for line in text_region.text_lines:
                    text_region_txt += line.text + "\n"

                text_region_list_of_dicts.append({"text_block_id": text_region_id, "text": text_region_txt})

            article_list_of_dicts.append({"article_id": article_id, "text_blocks": text_region_list_of_dicts})

        json_dict["page"].append({"page_file": page_name, "articles": article_list_of_dicts})

    json_object = json.dumps(json_dict, ensure_ascii=False, indent=None)

    # writing to json file
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
        logger.info(f"Dumped json {json_path}")
    logger.info(f"Number of region skips (missing article IDs) = {region_skips}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_paths", type=str, help="list file containing paths to pageXML files for GT generation",
                        required=True)
    parser.add_argument("--json_path", type=str, help="output path for GT json file", required=True)
    args = parser.parse_args()

    generate_finetuning_json(args.page_paths, args.json_path)
