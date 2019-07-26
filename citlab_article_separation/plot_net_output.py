from __future__ import print_function, division

import os
from argparse import ArgumentParser

import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


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


def plot_net_output(path_to_pb, path_to_img_lst, gpu_device="0", n_class=3):
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
                img_name = os.path.basename(path_to_img)
                img_name, ext = os.path.splitext(img_name)

                # ext = ".png"

                path_to_gt_0 = os.path.join(dirname, "C3", img_name + "_GT0" + ext)
                path_to_gt_1 = os.path.join(dirname, "C3", img_name + "_GT1" + ext)
                path_to_gt_2 = os.path.join(dirname, "C3", img_name + "_GT2" + ext)

                img = cv2.imread(path_to_img)
                gt0_img = cv2.imread(path_to_gt_0)
                gt1_img = cv2.imread(path_to_gt_1)
                gt2_img = cv2.imread(path_to_gt_2)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                if len(img_gray.shape) == 2:
                    img_gray = np.expand_dims(img_gray, axis=-1)
                    img_gray = np.expand_dims(img_gray, axis=0)
                out_np = sess.run(out, feed_dict={x: img_gray})
                out_img = out_np

                out_img_masked = np.ma.masked_where(out_img < 0.05, out_img)
                # out_img = np.array((out_img > 0.99), np.uint8)

                # fig = plt.figure()
                # plt.imshow(img, cmap=plt.cm.gray)
                # plt.imshow(out_img_masked[0, :, :, 0] * 255, cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
                # plt.savefig("./tests/resources/test_plot_net_output/tmp/" + img_name + "_GT0" + ext, dpi=2000)
                # # cv2.imwrite("./tests/resources/test_plot_net_output/tmp/" + img_name + "_GT0" + ext,
                # #                             out_img[0, :, :, 0] * 255)

                n_class_img = out_img.shape[-1]
                fig = plt.figure()
                a = fig.add_subplot(2, n_class_img + 2, 1)
                plt.imshow(img, cmap=plt.cm.gray)
                # plt.imshow(out_img[0, :, :, :], cmap=plt.cm.gray)
                a.set_title('input image')

                fig.add_subplot(2, n_class_img + 2, n_class_img + 4)
                plt.imshow(gt0_img)
                fig.add_subplot(2, n_class_img + 2, n_class_img + 5)
                plt.imshow(gt1_img)
                fig.add_subplot(2, n_class_img + 2, n_class_img + 6)
                plt.imshow(gt2_img)

                for cl in range(n_class_img):
                    a = fig.add_subplot(2, n_class_img + 2, cl + 2)
                    plt.imshow(out_img[0, :, :, cl], cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
                    cv2.imwrite("./tests/resources/test_plot_net_output/filled_articles/" + img_name + "_GT" + str(cl) + ext,
                                out_img[0, :, :, cl] * 255)
                    a.set_title('Channel: ' + str(cl))
                plt.show()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--path_to_tf_graph', default='', type=str,
                        help="path to the TensorFlow .pb file containing the ARU-Net graph.")
    parser.add_argument('--path_to_img_lst', default='', type=str,
                        help='path to the lst file containing the file paths of the images.')

    args = parser.parse_args()

    plot_net_output(args.path_to_tf_graph, args.path_to_img_lst)
