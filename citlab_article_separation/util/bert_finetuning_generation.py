import json
import os
import argparse
import logging
from citlab_python_util.parser.xml.page.page import Page


def generate_finetuning_json(page_paths, json_path):
    xml_files = [line.rstrip('\n') for line in open(page_paths, "r")]
    json_dict = {"page": []}

    num_errors = 0
    for xml_file in xml_files:
        # load the page xml files
        page_file = Page(xml_file)
        page_name = os.path.splitext(os.path.basename(xml_file))[0]
        logging.info(f"Processing {xml_file}")

        # load text regions
        list_of_txt_regions = page_file.get_text_regions()

        article_id_txt_region_id_dict = {}
        txt_region_id_txt_lines_dict = {}

        for txt_region in list_of_txt_regions:
            id_list = []

            for txt_line in txt_region.text_lines:
                id = txt_line.get_article_id()
                id_list.append(id)

            if len(set(id_list)) > 1:
                # num_errors += 1
                logging.warning(f"Textregion {txt_region.id} contains more than 1 article ID: {set(id_list)}. "
                                f"Skipping textregion.")
                continue
            else:
                # article id of the text region
                try:
                    article_id = id_list[0]
                except IndexError as ex:
                    num_errors += 1
                    logging.warning(f"Error indexing article IDs for textregion {txt_region.id}. {ex}. Skipping page.")
                    break
                if article_id not in article_id_txt_region_id_dict:
                    article_id_txt_region_id_dict.update({article_id: [txt_region.id]})
                else:
                    article_id_txt_region_id_dict[article_id].append(txt_region.id)
                txt_region_id_txt_lines_dict.update({txt_region.id: txt_region.text_lines})

        article_list_of_dicts = []
        for article_id in article_id_txt_region_id_dict:
            txt_region_list_of_dicts = []

            for txt_region_id in article_id_txt_region_id_dict[article_id]:
                txt_region_txt = ""

                for line in txt_region_id_txt_lines_dict[txt_region_id]:
                    txt_region_txt += line.text + "\n"

                txt_region_list_of_dicts.append({"text_block_id": txt_region_id, "text": txt_region_txt})

            article_list_of_dicts.append({"article_id": article_id, "text_blocks": txt_region_list_of_dicts})

        json_dict["page"].append({"page_file": page_name, "articles": article_list_of_dicts})

    json_object = json.dumps(json_dict, ensure_ascii=False, indent=None)

    # writing to json file
    with open(json_path, "w") as outfile:
        outfile.write(json_object)
        logging.info(f"Dumped json {json_path}")
    logging.info(f"Number of errors = {num_errors}")


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--page_paths", type=str, help="list file containing paths to pageXML files for GT generation",
                        required=True)
    parser.add_argument("--json_path", type=str, help="output path for GT json file", required=True)
    args = parser.parse_args()

    generate_finetuning_json(args.page_paths, args.json_path)
