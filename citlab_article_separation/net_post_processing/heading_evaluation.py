import argparse
import os

import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

from citlab_article_separation.net_post_processing.heading_net_post_processor import HeadingNetPostProcessor
from citlab_python_util.io.file_loader import load_list_file, get_page_path
from citlab_python_util.parser.xml.page.page import Page
from citlab_python_util.parser.xml.page.page_constants import TextRegionTypes


def get_heading_regions(page_object: Page):
    text_regions = page_object.get_text_regions()
    return [text_region for text_region in text_regions if text_region.region_type == TextRegionTypes.sHEADING]


def get_heading_text_lines(heading_regions):
    text_lines = []
    for heading_region in heading_regions:
        text_lines.extend(heading_region.text_lines)
    return text_lines


def get_heading_text_line_by_custom_type(heading_regions):
    text_lines = []
    for heading_region in heading_regions:
        from citlab_python_util.parser.xml.page.page_objects import TextLine
        text_line: TextLine
        for text_line in heading_region.text_lines:
            try:
                if text_line.custom["structure"]["semantic_type"] == TextRegionTypes.sHEADING:
                    text_lines.append(text_line)
            except KeyError:
                continue

    return text_lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_gt_list', type=str, required=True,
                        help='Path to the list of GT PAGE XML file paths.')
    parser.add_argument('--path_to_pb', type=str, required=True,
                        help="Path to the TensorFlow pb graph for creating the separator information")
    parser.add_argument('--fixed_height', type=int, required=False,
                        help="If parameter is given, the images will be scaled to this height by keeping the aspect "
                             "ratio")
    parser.add_argument('--threshold', type=float, required=False,
                        help="Threshold value that decides based on the feature values if a text line is a heading or "
                             "not.")
    parser.add_argument('--net_weight', type=float, required=False, help="Weight the net output feature.")
    parser.add_argument('--stroke_width_weight', type=float, required=False, help="Weight the stroke width feature.")
    parser.add_argument('--text_height_weight', type=float, required=False, help="Weight the text line height feature.")
    parser.add_argument('--gpu_devices', type=str, required=False,
                        help='Which GPU devices to use, comma-separated integers. E.g. "0,1,2".',
                        default='0')
    parser.add_argument('--log_file_folder', type=str, required=False, help='Where to store the log files.')

    args = parser.parse_args()

    path_to_gt_list = args.path_to_gt_list
    path_to_pb = args.path_to_pb
    fixed_height = args.fixed_height
    is_heading_threshold = args.threshold
    net_weight = args.net_weight
    stroke_width_weight = args.stroke_width_weight
    text_height_weight = args.text_height_weight
    log_file_folder = args.log_file_folder
    gpu_devices = args.gpu_devices

    # net_weight = 0.33
    # stroke_width_weight = 0.33
    # text_height_weight = 0.33

    weight_dict = {"net": net_weight,
                   "stroke_width": stroke_width_weight,
                   "text_height": text_height_weight}

    # path_to_gt_list = "/home/max/data/la/heading_detection/post_process_experiments/image_paths.lst"
    #
    # path_to_gt_list = '/home/max/data/la/heading_detection/post_process_experiments/dummy_image_paths.lst'
    # path_to_pb = "/home/max/data/la/heading_detection/post_process_experiments/HD_ru_3000_export_best_2020-09-22.pb"
    # fixed_height = 500
    # is_heading_threshold = 0.5

    image_paths = load_list_file(path_to_gt_list)

    post_processor = HeadingNetPostProcessor(path_to_gt_list, path_to_pb, fixed_height, scaling_factor=None,
                                             weight_dict=weight_dict, threshold=is_heading_threshold)
    page_objects_hyp = post_processor.run(gpu_devices)

    f1_scores, recall_scores, precision_scores = [], [], []

    log_file_name = f"{fixed_height:04}_{is_heading_threshold*100:03.0f}_{net_weight*100:03.0f}_" \
                    f"{stroke_width_weight*100:03.0f}_{text_height_weight*100:03.0f}.log"
    log_file_name = os.path.join(log_file_folder, log_file_name)

    with open(log_file_name, 'w') as log_file:
        log_file.write(f"fixed_height: {fixed_height}\n"
                       f"is_heading_threshold: {is_heading_threshold}\n"
                       f"net_weight: {net_weight}\n"
                       f"stroke_width_weight: {stroke_width_weight}\n"
                       f"text_height_weight: {text_height_weight}\n")
        for i, image_path in enumerate(image_paths):
            log_file.write(f"\nImage path: {image_path}\n")
            xml_path = get_page_path(image_path)
            page_object_gt = Page(xml_path)
            page_object_hyp = page_objects_hyp[i]

            text_regions_gt = page_object_gt.get_text_regions()
            text_regions_hyp = page_object_hyp.get_text_regions()

            is_heading_gt = [tr.region_type == TextRegionTypes.sHEADING for tr in text_regions_gt]
            is_heading_hyp = [tr.region_type == TextRegionTypes.sHEADING for tr in text_regions_hyp]

            recall_scores.append(recall_score(is_heading_gt, is_heading_hyp, average='binary'))
            precision_scores.append(precision_score(is_heading_gt, is_heading_hyp, average='binary'))
            f1_scores.append(f1_score(is_heading_gt, is_heading_hyp, average='binary'))

            log_file.write(f"\t{'R':>2}: {recall_scores[-1]:.4f}\n")
            log_file.write(f"\t{'P':>2}: {precision_scores[-1]:.4f}\n")
            log_file.write(f"\t{'F1':>2}: {f1_scores[-1]:.4f}\n")

        avg_recall = np.mean(recall_scores)
        avg_precision = np.mean(precision_scores)
        avg_f1 = np.mean(f1_scores)

        log_file.write("\nAverage Recall \t Average Precision \t Average F1\n")
        log_file.write(f"{avg_recall:.4f}, {avg_precision:.4f}, {avg_f1:.4f}")
