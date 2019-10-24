from __future__ import print_function, division

import os
from argparse import ArgumentParser

import cv2
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


def plot_net_output(path_to_pb, path_to_img_lst, save_folder="", gpu_device="0", rescale=None, mask_threshold=None,
                    plot_with_gt=False):
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
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                if rescale:
                    img_gray = cv2.resize(img_gray, None, fx=rescale, fy=rescale, interpolation=cv2.INTER_AREA)

                if len(img_gray.shape) == 2:
                    img_gray = np.expand_dims(img_gray, axis=-1)
                    img_gray = np.expand_dims(img_gray, axis=0)
                out_img = sess.run(out, feed_dict={x: img_gray})

                if mask_threshold:
                    out_img = np.array((out_img > 0.6), np.int32)

                n_class_img = out_img.shape[-1]

                if plot_with_gt:
                    paths_to_gts = [os.path.join(dirname, "C" + str(n_class_img), img_name + "_GT" + str(i) + ".png") for i in range(n_class_img)]
                    gt_imgs = [cv2.imread(path_to_gt) for path_to_gt in paths_to_gts]
                    gt_imgs = [cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) for gt_img in gt_imgs]
                    if rescale:
                        gt_imgs = [cv2.resize(gt_img, None, fx=rescale, fy=rescale) for gt_img in gt_imgs]

                for cl in range(n_class_img):
                    out_img_2d = out_img[0, :, :, cl]

                    if plot_with_gt:
                        gt_img = gt_imgs[cl]
                        out_img_2d = np.concatenate((out_img_2d, gt_img), axis=1)

                    cv2.imwrite(
                        os.path.join(save_folder, img_name + "_OUT" + str(cl) + ext), out_img_2d * 255)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_to_tf_graph', default='', type=str,
                        help="path to the TensorFlow .pb file containing the ARU-Net graph.")
    parser.add_argument('--path_to_img_lst', default='', type=str,
                        help='path to the lst file containing the file paths of the images.')
    parser.add_argument('--save_folder', default='', type=str, help="path to the save folder")
    parser.add_argument('--rescale_factor', default=1.0, type=float,
                        help="rescaling the images before inputting them to the network.")

    args = parser.parse_args()

    if not os.path.exists(args.save_folder) or not os.path.isdir(args.save_folder):
        os.mkdir(args.save_folder)

    plot_net_output(args.path_to_tf_graph, args.path_to_img_lst, args.save_folder, rescale=args.rescale_factor,
                    plot_with_gt=True, mask_threshold=False)
