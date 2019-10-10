# -*-coding:utf-8-*-

import numpy as np
import cv2
import os
from PIL import Image
import mxnet as mx
import time
import copy

from Apps.libs.s_model_predict.util import load_weights
from Apps.libs.s_model_predict import symbol

# from .seg_labels import kd_germany_test_labels as kd_road_deploy_labels

# from cls13_large_20180507.kd_helper import kd_road_deploy_labels
# from cls13_multilabel_20180528.kd_helper import kd_road_deploy_labels
# from models.arrow_classfication.model import ModelClassArrow
#   from models.virtual_lane.model import ModelVirtualLane

#   import global_variables


class ModelResNetRoad:

    def __init__(self, gpu_id=0, model_path=None, cls=None, seg_label_li=None, output_confidence=False, info=None):
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        cur_path = os.path.realpath(__file__)
        cur_dir = os.path.dirname(cur_path)
        cur_list = os.listdir(cur_dir)
        self.info = info
        self.cls = cls
        self.weights = model_path
        self.seg_label_li = seg_label_li
        self.ignore_color = (0, 0, 0)
        # clr_dict = {l.name: l.color for l in kd_road_deploy_labels}
        # for name, color in clr_dict.items():
        # for k, v in self.seg_label_li.items():
        #     if v[0] == u'Ignore':
        #         self.ignore_color = tuple(v[1])
        #         break
        for i in self.seg_label_li:
            if i['name'] == u'Ignore':
                self.ignore_color = tuple(i['color'])
                break

        # label to color
        self.use_label_to_color = True
        self.label_to_color = np.zeros((256, 3), dtype=np.uint8)
        # for k, v in self.seg_label_li.items():
        #     if not int(k) in range(0, 256):
        #         continue
        #     self.label_to_color[int(k)] = (
        #         int(v[1][2]), int(v[1][1]), int(v[1][0]))
        for label in self.seg_label_li:
            if not int(label["categoryId"]) in range(0, 256):
                continue
            self.label_to_color[int(label["categoryId"])] = (label["color"][2], label["color"][1], label["color"][0])
            if label["color"] == [128, 64, 128]:
                self.road_id = int(label["categoryId"])
            elif label["color"] == [64, 64, 32]:
                self.other_id = int(label["categoryId"])

        network, net_args, net_auxs = load_weights(self.weights)
        context = [mx.gpu(gpu_id)]
        self.mod = mx.mod.Module(network, context=context)

        self.result_shape = [1024, 2448]
        self.input_shape = [512, 1224]
        self.batch_data_shape = (1, 3, 512, 1224)
        provide_data = [("data", self.batch_data_shape)]
        self.batch_label_shape = (1, 512, 1224)
        provide_label = [("softmax_label", self.batch_label_shape)]
        self.mod.bind(provide_data, provide_label, for_training=False, force_rebind=True)
        self.mod.init_params(arg_params=net_args, aux_params=net_auxs)
        self._flipping = False

        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        symbol.cfg['workspace'] = 1024
        symbol.cfg['bn_use_global_stats'] = True

        # GPU Upsampling
        self.add_conf = output_confidence
        self.use_gpu_upsampling = True
        upsampling_sym, confidence_sym = self.get_upsampling_sym()
        self.upsample_mod = mx.mod.Module(upsampling_sym, context=context, data_names=['data'], label_names=[])
        if self.add_conf:
            self.confidence_mod = mx.mod.Module(confidence_sym, context=context, data_names=['data'], label_names=[])

        self.upsample_mod.bind(
            data_shapes=[('data', (1, int(self.cls), 512, 1224))],
            label_shapes=None,
            for_training=False,
            force_rebind=True)
        if self.add_conf:
            self.confidence_mod.bind(
                data_shapes=[('data', (1, int(self.cls), 512, 1224))],
                label_shapes=None,
                for_training=False,
                force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)
        if self.add_conf:
            self.confidence_mod.init_params(initializer=initializer)

    # self.virtual_lane_net = ModelVirtualLane(gpu_id=gpu_id)

    def get_upsampling_sym(self):
        input_data = mx.symbol.Variable(name='data')
        upsampling = mx.symbol.UpSampling(
            input_data,
            scale=2,
            num_filter=int(self.cls),
            sample_type='bilinear',
            name="upsampling_preds",
            workspace=512)
        upsampling_sym = mx.sym.argmax(data=upsampling, axis=1)
        confidence_sym = mx.sym.max(data=upsampling, axis=1)
        return upsampling_sym, confidence_sym

    def do(self, image_data, dest_file=None, info_queue=None):
        pred_data = None
        try:
            _time1 = time.time()
            image = np.asarray(bytearray(image_data), dtype="uint8")
            origin_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)

            # read image as rgb
            origin_frame = origin_frame[:, :, ::-1]
            width = origin_frame.shape[1]
            height = origin_frame.shape[0]

            # crop bottom half of the picture
            bottom_half = origin_frame[height - self.result_shape[0]:height, 0:width]

            img = np.array(
                Image.fromarray(bottom_half.astype(np.uint8, copy=False)).resize(
                    (self.input_shape[1], self.input_shape[0]), Image.NEAREST))
            # img = np.array(Image.fromarray(bottom_half.astype(np.uint8, copy=False)))

            img = img.transpose(2, 0, 1)
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

            self.upsample_mod.forward(mx.io.DataBatch(data=preds), is_train=False)
            out_pred = self.upsample_mod.get_outputs()[0].asnumpy().squeeze().astype(np.uint8)
            copy_out = copy.deepcopy(out_pred)
            _time1 = time.time()
            rgb_frame = self.label_to_color[out_pred]
            _time2 = time.time()
            print("id to color use:{} s".format(_time2 - _time1))

            recog_image = np.zeros((height, width, 3), np.uint8)
            recog_image[height - self.result_shape[0]:height, 0:width] = rgb_frame

            # img_array = cv2.imencode('.png', rgb_frame)
            # img_data = img_array[1]
            # pred_data = img_data.tostring()

            blank_image = np.zeros((height, width, 4), np.uint8)
            blank_image[0:height, 0:width] = (0, 0, 0, 255)

            blank_image[0:height - self.result_shape[0], 0:width] = (0, 0, 0, 0)
            blank_image[0:height, 0:width, 0] = recog_image[:, :, 0]
            blank_image[0:height, 0:width, 1] = recog_image[:, :, 1]
            blank_image[0:height, 0:width, 2] = recog_image[:, :, 2]

            if self.add_conf:
                self.confidence_mod.forward(mx.io.DataBatch(data=preds), is_train=False)
                outputs = self.confidence_mod.get_outputs()[0]
                confidence = (outputs.asnumpy().astype(np.single))
                confidence = confidence[0]
                copy_con = copy.deepcopy(confidence)

                confidence = np.where(np.logical_and(confidence > 0, confidence <= 0.05), 0, confidence)
                confidence = np.where(np.logical_and(confidence > 0.05, confidence <= 0.15), 0.1, confidence)
                confidence = np.where(np.logical_and(confidence > 0.15, confidence <= 0.25), 0.2, confidence)
                confidence = np.where(np.logical_and(confidence > 0.25, confidence <= 0.35), 0.3, confidence)
                confidence = np.where(np.logical_and(confidence > 0.35, confidence <= 0.45), 0.4, confidence)
                confidence = np.where(np.logical_and(confidence > 0.45, confidence <= 0.55), 0.5, confidence)
                confidence = np.where(np.logical_and(confidence > 0.55, confidence <= 0.65), 0.6, confidence)
                confidence = np.where(np.logical_and(confidence > 0.65, confidence <= 0.75), 0.7, confidence)
                confidence = np.where(np.logical_and(confidence > 0.75, confidence <= 0.85), 0.8, confidence)
                confidence = np.where(np.logical_and(confidence > 0.85, confidence <= 0.95), 0.9, confidence)
                confidence = np.where(np.logical_and(confidence > 0.95, confidence <= 1), 1, confidence)
                img_conf = (confidence * 255).astype(np.uint8)
                blank_image[height - self.result_shape[0]:height, 0:width, 3] = img_conf

                a = np.where((blank_image[:, :, 0] == 128) &
                             (blank_image[:, :, 1] == 64) &
                             (blank_image[:, :, 2] == 128))
                blank_image[a] = [128, 64, 128, 255]

                b = np.where((blank_image[:, :, 0] == 32) &
                             (blank_image[:, :, 1] == 64) &
                             (blank_image[:, :, 2] == 64))
                blank_image[b] = [32, 64, 64, 255]

                whole_con = round(np.mean(copy_con), 4)
                c = np.where(copy_out[:] == self.road_id)
                d = np.where(copy_out[:] == self.other_id)
                do_whole_con = round((np.sum(copy_con) - np.sum(copy_con[c]) - np.sum(copy_con[d])) /
                                     (2448 * 1024 - len(c[0]) - len(d[0])), 4)

                cls = self.cls
                s_pixel = np.zeros(cls)
                bins = [i for i in range(cls + 1)]
                (n, _) = np.histogram(out_pred, bins=bins, normed=False)

                s_pixel += n
                o_cls_con = {}
                # cls_con = []
                for i in range(cls):
                    c = np.where((copy_out[:] == i))
                    # d = np.where((copy_out[:] == i))

                    o_cls_con["{}".format(i)] = str(round((np.sum(copy_con[c]) / s_pixel[i]), 4))
                    # cls_con.append({i: round((np.sum(confidence[c]) / s_pixel[i]), 4)})
                # model_info = self.weights.split('/')
                dest_name = os.path.basename(dest_file)
                info = self.info(whole_con, do_whole_con, o_cls_con, dest_name[:-11])
                info_queue.put(info)
            pred_data = blank_image

            if dest_file is not None:
                with open(dest_file, "wb") as f:
                    img_array = cv2.imencode('.png', pred_data)
                    img_data = img_array[1]
                    pred_data = img_data.tostring()
                    f.write(pred_data)
            # dest_path = os.path.join(os.path.dirname(dest_file), os.path.basename(dest_file)[:-4]+"-vir.png")
            # cv2.imwrite(dest_path, rgb_frame)

            return pred_data
        except Exception as e:
            print("recognition error:{}".format(repr(e)))
        finally:
            return pred_data


class Task:

    def __init__(self, img_data, dest_path, exit_flag=False):
        self.img_data = img_data
        self.dest_path = dest_path
        self.exit_flag = exit_flag


class Info:

    def __init__(self, origin_w_con, w_con, o_cls_con, trackpointid, exit_flag=False):
        self.or_w_con = origin_w_con
        self.w_con = w_con
        self.o_cls_con = o_cls_con
        # self.cls_con = cls_con
        # self.model = model
        self.trackpointid = trackpointid
        self.exit_flag = exit_flag


def do_seg(gpu_id, task_queue, model_path, count, cls, seg_label_li, lock, output_confidence, info_queue):
    info = Info
    net = ModelResNetRoad(
        gpu_id=gpu_id,
        model_path=model_path,
        cls=cls,
        seg_label_li=seg_label_li,
        output_confidence=output_confidence,
        info=info)
    while True:
        if task_queue.empty():
            break

        task = task_queue.get()

        if not isinstance(task, Task):
            break

        if task.exit_flag:
            break

        img_data = task.img_data
        dest_path = task.dest_path

        try:
            start = time.time()
            net.do(image_data=img_data, dest_file=dest_path, info_queue=info_queue)
            lock.acquire()
            count.value += 1
            end = time.time()
            lock.release()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))
