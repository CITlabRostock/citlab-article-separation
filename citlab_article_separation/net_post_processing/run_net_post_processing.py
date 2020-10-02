import argparse
from concurrent.futures import ProcessPoolExecutor

from citlab_article_separation.net_post_processing.separator_net_post_processor import SeparatorNetPostProcessor
from citlab_python_util.io.file_loader import load_list_file


def run(image_list, path_to_pb, fixed_height, scaling_factor, threshold):
    post_processor = SeparatorNetPostProcessor(image_list, path_to_pb, fixed_height, scaling_factor, threshold,
                                               gpu_devices='')
    post_processor.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_image_list", type=str, required=True,
                        help="Path to the list file holding the image paths.")
    parser.add_argument("--path_to_pb", type=str, required=True,
                        help="Path to the TensorFlow pixel labelling graph.")
    parser.add_argument("--num_processes", type=int, required=False,
                        help="Number of processes that run in parallel.", default=8)
    parser.add_argument("--fixed_height", type=int, required=False,
                        help="Input image height", default=1500)
    parser.add_argument("--scaling_factor", type=float, required=False,
                        help="Scaling factor of images.", default=1.0)
    parser.add_argument("--threshold", type=float, required=False,
                        help="Threshold for binarization of net output.", default=0.05)

    args = parser.parse_args()

    image_path_list = load_list_file(args.path_to_image_list)
    path_to_pb = args.path_to_pb
    num_processes = args.num_processes

    fixed_height = args.fixed_height
    scaling_factor = args.scaling_factor
    threshold = args.threshold

    MAX_SUBLIST_SIZE = 50

    with ProcessPoolExecutor(num_processes) as executor:
        size_sub_lists = len(image_path_list) // num_processes
        if size_sub_lists == 0:
            size_sub_lists = 1
            num_processes = len(image_path_list)
        size_sub_lists = min(MAX_SUBLIST_SIZE, size_sub_lists)

        image_path_sub_lists = [image_path_list[i: i + size_sub_lists] for i in
                                range(0, len(image_path_list), size_sub_lists)]

        run_args = ((image_path_sub_list, path_to_pb, fixed_height, scaling_factor, threshold) for image_path_sub_list
                    in image_path_sub_lists)

        [executor.submit(run, *run_arg) for run_arg in run_args]
