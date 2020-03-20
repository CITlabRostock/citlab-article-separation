from __future__ import print_function, division

import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we can use again a convenient built-in function to import a graph_def into the
    # current default Graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def plot_image_with_net_output(image, net_output):
    net_output_rgb_int = np.uint8(cv2.cvtColor(net_output, cv2.COLOR_GRAY2BGR))
    net_output_rgb_int = cv2.cvtColor(net_output_rgb_int, cv2.COLOR_BGR2HLS)

    res = cv2.addWeighted(image, 0.9, net_output_rgb_int, 0.4, 0)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    return res


def plot_connected_components(image):
    _, image_bin = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(255 - image_bin)

    # res = cv2.addWeighted(image, 0.4, labels, 0.1, 0, dtype=2)
    return labels


def plot_net_output(path_to_pb, path_to_img_lst, save_folder="", gpu_device="0", rescale=None, fixed_height=None,
                    mask_threshold=None, plot_with_gt=False, plot_with_img=False, show_plot=False):
    session_conf = tf.ConfigProto()
    session_conf.gpu_options.visible_device_list = gpu_device

    graph = load_graph(path_to_pb)

    with tf.Session(graph=graph, config=session_conf) as sess:
        x = graph.get_tensor_by_name('inImg:0')
        out = graph.get_tensor_by_name('output:0')

        with open(path_to_img_lst) as f:
            for path_to_img in f.readlines():
                path_to_img = path_to_img.rstrip()

                dirname = os.path.dirname(path_to_img)
                img_name, ext = os.path.splitext(os.path.basename(path_to_img))

                img = cv2.imread(path_to_img)
                img_height, img_width = img.shape[:2]
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                scaling_factor = None
                if fixed_height and rescale and rescale != 1:
                    scaling_factor = rescale * fixed_height / img_height
                elif fixed_height:
                    scaling_factor = fixed_height / img_height

                if scaling_factor:
                    img = cv2.resize(img, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                    img_gray = cv2.resize(img_gray, None, fx=scaling_factor, fy=scaling_factor,
                                          interpolation=cv2.INTER_AREA)

                if len(img_gray.shape) == 2:
                    img_gray = np.expand_dims(img_gray, axis=-1)
                    img_gray = np.expand_dims(img_gray, axis=0)
                out_img = sess.run(out, feed_dict={x: img_gray})

                if mask_threshold:
                    out_img = np.array((out_img > 0.6), np.int32)

                n_class_img = out_img.shape[-1]

                if plot_with_gt:
                    paths_to_gts = [os.path.join(dirname, "C" + str(n_class_img), img_name + "_GT" + str(i) + ".png")
                                    for i in range(n_class_img)]
                    gt_imgs = [cv2.imread(path_to_gt) for path_to_gt in paths_to_gts]
                    gt_imgs = [cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) for gt_img in gt_imgs]
                    if scaling_factor:
                        gt_imgs = [cv2.resize(gt_img, None, fx=scaling_factor, fy=scaling_factor) for gt_img in gt_imgs]

                for cl in range(n_class_img):
                    out_img_2d = out_img[0, :, :, cl]
                    out_img_2d_255 = out_img_2d * 255
                    out_img_2d_255 = np.uint8(out_img_2d_255)

                    if plot_with_img:
                        out_img_2d_255 = plot_image_with_net_output(img, out_img_2d_255)
                    if plot_with_gt:
                        gt_img = gt_imgs[cl]
                        if plot_with_img:
                            gt_img = plot_image_with_net_output(img, gt_img)
                        out_img_2d_255 = np.concatenate((out_img_2d_255, gt_img), axis=1)
                    if save_folder:
                        cv2.imwrite(
                            os.path.join(save_folder, img_name + "_OUT" + str(cl) + ext), out_img_2d_255)
                    if show_plot:
                        if plot_with_img:
                            plt.imshow(out_img_2d_255)
                        else:
                            plt.imshow(out_img_2d_255, cmap="gray")
                        img_gray_2d = img_gray[0, :, :, 0]
                        #
                        # img_with_net_output = plot_image_with_net_output(img, out_img_2d_255)
                        # plt.imshow(img_with_net_output, cmap='gray')
                        # plt.imshow(img)
                        # print(img.shape)
                        # plt.show()
                        # plt.imshow(img_with_net_output)
                        img_with_ccs = plot_connected_components(img_gray_2d)
                        plt.imshow(img_with_ccs, cmap='gray')
                        plt.show()
                        exit(1)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_to_tf_graph', default='', type=str,
                        help="path to the TensorFlow .pb file containing the ARU-Net graph.")
    parser.add_argument('--path_to_img_lst', default='', type=str,
                        help='path to the lst file containing the file paths of the images.')
    parser.add_argument('--save_folder', default='', type=str, help="path to the save folder")
    parser.add_argument('--rescale_factor', default=1.0, type=float,
                        help="rescaling the images before inputting them to the network.")
    parser.add_argument('--fixed_height', default=0, type=int,
                        help="rescales the input images to a fixed height. Only works if rescale_factor is not set")

    args = parser.parse_args()

    if not os.path.exists(args.save_folder) or not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    plot_net_output(args.path_to_tf_graph, args.path_to_img_lst, args.save_folder, rescale=args.rescale_factor,
                    fixed_height=args.fixed_height, plot_with_gt=False, plot_with_img=True,
                    mask_threshold=False, show_plot=False)
