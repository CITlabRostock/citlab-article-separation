import time
import concurrent.futures
from multiprocessing import cpu_count
import os
import argparse

def run_grid_search(fixed_height, threshold, net_weight):
    net_weight_f = net_weight / 10
    for stroke_width_weight in range(0, 10 - net_weight + 1, 1):
        stroke_width_weight_f = stroke_width_weight / 10
        text_height_weight_f = (10 - net_weight - stroke_width_weight) / 10

        os.system("python -u /home/max/devel/src/git/python/citlab-article-separation/citlab_article_separation/net_post_processing/heading_evaluation.py --path_to_gt_list {} --path_to_pb {} --fixed_height {} --threshold {} --net_weight {} --stroke_width_weight {} --text_height_weight {} --gpu_devices '' --log_file_folder {}".format(PATH_TO_GT_LIST, PATH_TO_PB, fixed_height, threshold, net_weight_f, stroke_width_weight_f, text_height_weight_f, LOG_FILE_FOLDER))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_gt_list", type=str, required=True,
                        help='Path to GT image list',
                        default='/home/max/data/la/heading_detection/post_process_experiments/image_paths_dummy.lst')
    parser.add_argument("--path_to_pb", type=str, required=True,
                        help='Path to the TensorFlow graph.',
                        default='/home/max/data/la/heading_detection/post_process_experiments/HD_ru_3000_export_best_2020-09-22.pb')
    parser.add_argument("--log_file_folder", type=str, required=True,
                        help='Path to the folder where the log files are stored',
                        default='/home/max/tests/logs')
    cmd_args = parser.parse_args()
    PATH_TO_GT_LIST = cmd_args.path_to_gt_list
    PATH_TO_PB = cmd_args.path_to_pb
    LOG_FILE_FOLDER = cmd_args.log_file_folder

    num_processes = cpu_count() // 2
    with concurrent.futures.ProcessPoolExecutor(num_processes) as executor:
        fixed_heights = range(600, 1300, 100)
        thresholds = range(4, 10, 1)
        net_weights = range(0, 11, 1)
        args = ((f, t/10, nw) for f in fixed_heights for t in thresholds for nw in net_weights)
        [executor.submit(run_grid_search, *arg) for arg in args]


