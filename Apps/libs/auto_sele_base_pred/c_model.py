# -*-coding:utf-8-*-

import argparse
import copy
import os
import time

import cv2
import mxnet as mx
import numpy as np
from PIL import Image

import m_symbol
from m_util import load_weights


class ModelResNetRoad:
    def __init__(self, weights_dir, gpu_id=0):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        # self.weights = os.path.join(cur_dir, 'kddata_cls13_s1_focal_ep-0240.params')
        self.weights = weights_dir
        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.result_shape = [1024, 2448]
        self.input_shape = [512, 1224]
        # self.batch_data_shape = (1, 3, 1024, 2448)
        self.batch_data_shape = (1, 3, 512, 1224)
        provide_data = [("data", self.batch_data_shape)]
        self.batch_label_shape = (1, 512, 1224)
        provide_label = [("softmax_label", self.batch_label_shape)]
        self.mod.bind(provide_data,
                      provide_label,
                      for_training=False,
                      force_rebind=True)
        self.mod.init_params(arg_params=net_args,
                             aux_params=net_auxs)
        self._flipping = False

        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        m_symbol.cfg['workspace'] = 1024
        m_symbol.cfg['bn_use_global_stats'] = True

        # GPU Upsampling
        self.use_gpu_upsampling = True
        upsampling_sym, confidence_sym = self.get_upsampling_sym()
        self.upsample_mod = mx.mod.Module(upsampling_sym,
                                          context=context,
                                          data_names=['data'],
                                          label_names=[])
        self.upsample_mod.bind(data_shapes=[('data', (1, 13, 512, 1224))],
                               label_shapes=None,
                               for_training=False,
                               force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)

        self.confidence_mod = mx.mod.Module(confidence_sym, context=context,
                                            data_names=['data'],
                                            label_names=[])
        self.confidence_mod.bind(data_shapes=[('data', (1, 12L, 256L, 612L))],
                                 label_shapes=None,
                                 for_training=False,
                                 force_rebind=True)
        self.confidence_mod.init_params(initializer=initializer)

    def cut_img(self, img_file):
        out = os.path.dirname(img_file)
        img = cv2.imread(img_file)
        h, w = img.shape[:2]
        h = h // 2
        img1 = img[h:, :]
        save_file = os.path.join(out, 'cut-' + os.path.basename(img_file))
        cv2.imwrite(save_file, img1)
        return save_file

    def read_image(self, img_file):
        img = Image.open(img_file)
        w, h = img.size
        if h > 1024:
            img_path = self.cut_img(img_file)
            img = Image.open(img_path)
            w, h = img.size
        w = w // 2
        h = h // 2
        img = img.resize((w, h))
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        return img

    def get_upsampling_sym(self):
        input_data = mx.symbol.Variable(name='data')
        upsampling = mx.symbol.UpSampling(input_data,
                                          scale=2,
                                          num_filter=13,
                                          sample_type='bilinear',
                                          name="upsampling_preds",
                                          workspace=512)
        upsampling_sym = mx.sym.argmax(data=input_data, axis=1)
        confidence_sym = mx.sym.max(data=input_data, axis=1)
        return upsampling_sym, confidence_sym

    def do(self, image_file):
        pred_label = None
        confidence = None
        try:

            img = self.read_image(image_file)
            self.batch_data[0][0] = img

            self.mod.forward(mx.io.DataBatch(data=self.batch_data, label=self.batch_label), is_train=False)

            if self._flipping:
                preds = copy.deepcopy(self.mod.get_outputs())
                flip_batch_data = []
                for batch_split_data in self.batch_data:
                    flip_batch_data.append(mx.nd.array(batch_split_data.asnumpy()[:, :, :, ::-1]))
                self.mod.forward(mx.io.DataBatch(flip_batch_data, label=self.batch_label), is_train=False)
                flip_preds = self.mod.get_outputs()
                merge_preds = []
                for i, pred in enumerate(preds):
                    # change left-lane and right-lane dimension when flipplig
                    flipped_pred = flip_preds[i].asnumpy()[:, :, :, ::-1]
                    flipped_pred[:, [1, 2], :, :] = flipped_pred[:, [2, 1], :, :]
                    merge_preds.append(mx.nd.array((0.5 * pred.asnumpy() + 0.5 * flipped_pred)))
                preds = merge_preds
            else:
                preds = self.mod.get_outputs()

            self.confidence_mod.forward(mx.io.DataBatch(data=preds), is_train=False)
            outputs = self.confidence_mod.get_outputs()[0]
            confidence = (outputs.asnumpy().astype(np.single))
            self.upsample_mod.forward(mx.io.DataBatch(data=preds), is_train=False)
            pred_label = self.upsample_mod.get_outputs()[0].asnumpy().squeeze().astype(np.uint8)
            return pred_label, confidence
        except Exception as e:
            print("recognition error:{}".format(repr(e)))
        finally:
            return pred_label, confidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True)
    args = parser.parse_args()

    image_dir = args.dir
    model_net = ModelResNetRoad(gpu_id=0)

    file_list = os.listdir(image_dir)
    for id_ in file_list:
        name_list = str(id_).split(".")
        if len(name_list) != 2:
            continue

        name_only = name_list[0]
        name_ext = name_list[1]
        if name_ext != 'png' and name_ext != 'jpg':
            continue

        file_path = os.path.join(image_dir, id_)
        dest_dir = os.path.join(image_dir, "resnet_road")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        file_id = name_only + ".png"
        dest_path = os.path.join(dest_dir, file_id)

        if os.path.exists(dest_path):
            continue

        try:
            start = time.time()
            with open(file_path, "rb") as f:
                img = f.read()
                model_net.do(img)
            end = time.time()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))
