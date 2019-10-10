# -*-coding:utf-8-*-
import sys

sys.path.insert(0, "/opt/densenet.mxnet")
sys.path.insert(0, "/opt/densenet.mxnet")

import numpy as np
import os
import mxnet as mx
import time
import cv2
from collections import namedtuple
from util import load_weights

# define a simple data batch
Batch = namedtuple('Batch', ['data'])


class ModelClassArrow(object):
    def __init__(self, model_file, gpu_id=9):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)

        # model_file_name = "densenet-kd-169-0-5000.params"
        self.weights = model_file

        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.input_shape = [224, 224]  # (W, H)
        # self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
        #               label_shapes=[('softmax_label', (1,))])
        self.mod.bind(for_training=False, data_shapes=[('data', (1, 3, self.input_shape[1], self.input_shape[0]))],
                      label_shapes=None)
        self.mod.init_params(arg_params=net_args,
                             aux_params=net_auxs)
        self._flipping = False

    def do(self, count, lock, image_data):
        pred_data = None
        accuracy = 0
        try:
            # time1
            time1 = time.time()
            image = np.asarray(bytearray(image_data), dtype="uint8")
            img = cv2.imdecode(image, cv2.IMREAD_COLOR)
            # print "original img size:", img.shape
            print("load img  time: {}".format(time.time() - time1))
            # pad image
            # img = np.array(Image.fromarray(origin_frame.astype(np.uint8, copy=False)))
            time2 = time.time()
            newsize = max(img.shape[:2])
            new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
            margin0 = (newsize - img.shape[0]) // 2
            margin1 = (newsize - img.shape[1]) // 2
            new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
            # img: (256, 256, 3), GBR format, HWC
            img = cv2.resize(new_img, tuple(self.input_shape))
            print("resize  time:  {}".format(time.time() - time2))
            # print "resized img size:", img.shape
            time3 = time.time()
            img = img.transpose(2, 0, 1)
            img = img[np.newaxis, :]

            # compute the predict probabilities
            self.mod.forward(Batch([mx.nd.array(img)]))
            prob = self.mod.get_outputs()[0].asnumpy()

            # Return the top-5
            prob = np.squeeze(prob)
            acc = np.sort(prob)[::-1]
            a = np.argsort(prob)[::-1]
            # forwordtime1
            print("forword time:  {}".format(time.time() - time3))

            # result = []
            # for i in a[0:5]:
            #     result.append((prob[i].split(" ", 1)[1], round(prob[i], 3)))

            pred_data = a[0:5]
            accuracy = acc[0:5]
            lock.acquire()
            count.value += 1
            lock.release()
        except Exception as e:
            print("recognition error:{}".format(repr(e)))

        return pred_data, accuracy
