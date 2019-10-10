#!/usr/bin/env python
#encoding=utf-8

import sys
import os
import time
import multiprocessing
import logging
import numpy as np
import shutil
import argparse
from PIL import Image
from Apps.libs.tensor import utils_2d
from Apps.libs.tensor import val_metric_v2_trt

import pycuda.driver as cuda
import tensorrt as trt

# os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = '0'


class Profiler(trt.infer.Profiler):

    def __init__(self, timing_iter):
        trt.infer.Profiler.__init__(self)
        self.timing_iterations = timing_iter
        self.profile = []

    def __del__(self):
        print "__del__"

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:-<70.70} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.2f} ms per iteration".format(totalTime / self.timing_iterations))

    def reset(self):
        self.profile = []


class TrtInfer(object):

    def __init__(self,
                 plan_file,
                 input_shape,
                 output_shape,
                 input_data_type=np.float32,
                 output_data_type=np.float32,
                 argmax=True,
                 effective_c=0,
                 gpuid=0,
                 debug_time=False):
        """
        plan_file: plan file
        input_shape: NCHW, must be same with the shape of plan file
        output_shape: NCHW, must match with the output shape of plan file
        input_data_type: default float32
        output_data_type: default float32
        argmax: default True
        effective_c: used in IN8 models
        gpuid: gpu index
        debug_time: display layer time
        """
        if not os.path.exists(plan_file):
            raise Exception("plan file does not exist: {}".format(plan_file))
        self.batch_size, self.input_c, self.input_h, self.input_w = input_shape
        output_batch_size, self.output_c, self.output_h, self.output_w = output_shape
        assert (self.batch_size == output_batch_size)
        self.output = np.empty(self.batch_size * self.output_c * self.output_w * self.output_h, dtype=output_data_type)
        self.temp_input = np.empty(1, dtype=input_data_type)
        self.argmax = argmax
        self.effective_c = effective_c
        self.debug_time = debug_time

        try:
            from importlib import import_module
            self.cuda = import_module("pycuda.driver")
            self.cuda.init()
        except ImportError as err:
            raise ImportError(
                """ERROR: Failed to import module({}) Please make sure you have pycuda and the example dependencies installed.
                                sudo apt-get install python(3)-pycuda
                                pip install tensorrt[examples]""".format(err))

        self.device = self.cuda.Device(gpuid)
        self.cuda_context = self.device.make_context()
        # Push the context - and make sure to pop it before returning!
        self.cuda_context.push()

        G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)
        self.engine = trt.utils.load_engine(G_LOGGER, plan_file)
        assert (self.engine)
        assert (self.engine.get_nb_bindings() == 2)
        self.context = self.engine.create_execution_context()

        self.d_input = cuda.mem_alloc(
            self.batch_size * self.input_c * self.input_w * self.input_h * self.temp_input.dtype.itemsize)
        self.d_output = cuda.mem_alloc(
            self.batch_size * self.output_c * self.output_w * self.output_h * self.output.dtype.itemsize)
        self.bindings = [int(self.d_input), int(self.d_output)]
        if (self.debug_time):
            self.g_prof = Profiler(1)
            self.context.set_profiler(self.g_prof)

        # Remove this from the context stack so Lite Engine is self contained.
        self.cuda_context.pop()

    def __del__(self):
        if self.cuda_context:
            # Must remove this context on destruction.
            self.cuda_context.pop()
        if self.context:
            self.context.destroy()
        if self.engine:
            self.engine.destroy()

    def destroy(self):
        self.__del__()

    def get_layer_time(self):
        if (self.debug_time):
            self.g_prof.print_layer_times()
        # else:
        #     raise("debug_time param is set False")

    def infer(self, data):
        if (self.debug_time):
            self.g_prof.reset()

        self.cuda_context.push()
        stream = self.cuda.Stream()
        cuda.memcpy_htod_async(self.d_input, data, stream)
        stream.synchronize()
        self.context.enqueue(self.batch_size, self.bindings, stream.handle, None)
        stream.synchronize()
        cuda.memcpy_dtoh_async(self.output, self.d_output, stream)
        stream.synchronize()
        self.cuda_context.pop()
        print "output value: [{},{}]".format(np.min(self.output), np.max(self.output))
        if (self.argmax):
            if (self.effective_c > 0):
                result = self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w)
                return [np.argmax(result[:, 0:self.effective_c, :, :], axis=1)]
            else:
                return [
                    np.argmax(
                        self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w), axis=1)
                ]
        else:
            return [self.output.reshape(self.batch_size, self.output_c, self.output_h, self.output_w)]


class EvalData(object):

    def __init__(self,
                 sour_dir,
                 output_dir,
                 weights_file,
                 input_size,
                 output_size,
                 count,
                 effective_c=0,
                 gpu_id=0,
                 batch_size=1,
                 height=1024,
                 width=1224,
                 use_half_image=False,
                 save_result_or_not=True,
                 iou_thresh_low=0.5,
                 min_pixel_num=2000,
                 flip=False,
                 upsample_or_not=False,
                 stride=1,
                 seg_label_li=None,
                 label_map_data=None):
        self.weights_file = weights_file
        self.output_dir = output_dir
        self.sour_dir = sour_dir
        self.label_map_data = label_map_data
        self.model_desc_file = os.path.join(self.output_dir, 'trt_info.txt')
        self.seg_label_li = seg_label_li
        self.use_half_image = use_half_image
        self.save_result_or_not = save_result_or_not
        self.count = count
        # Thresholds
        self.iou_thresh_low = iou_thresh_low
        self.min_pixel_num = min_pixel_num

        # Set batch size
        self.batch_size = batch_size

        # number of classes

        self.flip = flip
        self.upsample_or_not = upsample_or_not
        self.stride = stride

        # Module
        self.batch_data_shape = (self.batch_size, 3, height, width)
        provide_data = [("data", self.batch_data_shape)]
        self.batch_label_shape = (self.batch_size, height // stride, width // stride)
        provide_label = [("softmax_label", self.batch_label_shape)]

        # batch data & batch label
        self.batch_data = [np.empty(info[1]) for info in provide_data]
        if self.upsample_or_not:
            batch_label_2_shape = (self.batch_size, height, width)
            provide_label_2 = [("softmax_label", batch_label_2_shape)]
            self.batch_label = [np.empty(info[1]) for info in provide_label_2]
        else:
            self.batch_label = [np.empty(info[1]) for info in provide_label]
        # color map
        self.label_id_to_name = {}
        for kd_label in self.seg_label_li:
            label_id = kd_label['categoryId']
            if int(label_id) == 255 or int(label_id) == 254:
                continue
            if not label_id in self.label_id_to_name:
                self.label_id_to_name[label_id] = (kd_label['name'], kd_label['id'], kd_label['color'])
        self.num_classes = len(self.label_id_to_name)
        self.output_size = (1, self.num_classes, output_size[0], output_size[1])
        # metric
        self.seg_all_metric = val_metric_v2_trt.SegMetric()
        self.seg_all_metric.set_nclass(self.num_classes)
        self.seg_all_metric.reset()

        self.seg_metric = val_metric_v2_trt.SegMetric()
        self.seg_metric.set_nclass(self.num_classes)
        self.seg_metric.reset()

        self.trt_infer = TrtInfer(
            self.weights_file, input_size, self.output_size, gpuid=gpu_id, debug_time=False, effective_c=effective_c)

        self.number_of_samplers = 0
        logging.info("num_classes: {}".format(self.num_classes))

    def __del__(self):
        pass

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # doubt_file = os.path.join(self.output_dir, "doubt.list")
        # out_f = open(doubt_file, "w")
        image_file_list = []
        sour = "/data/deeplearning/train_platform/trt_eval_sour/s_data"
        if sour in self.sour_dir:
            img_li = os.listdir(sour)
        for i in img_li:
            if i.endswith('.jpg'):
                image_file_list.append(os.path.join(sour, i))
        total_pure_time = 0
        tic_all = time.time()
        cursor = 0
        number_of_samplers = len(image_file_list)
        self.number_of_samplers = number_of_samplers
        while cursor < number_of_samplers:
            try:
                cursor_to = min(number_of_samplers, cursor + self.batch_size)
                if not self.feed_batch_data(image_file_list, cursor, cursor_to):
                    cursor += self.batch_size
                    continue

                # Predicting
                self.batch_data[0] = self.batch_data[0].astype(np.float32)
                print "input type: {}".format(type(self.batch_data[0]))
                print "input shape: {}".format(self.batch_data[0].shape)
                tic = time.time()
                pred_label = self.trt_infer.infer(self.batch_data[0])
                self.trt_infer.get_layer_time()
                total_pure_time = total_pure_time + (time.time() - tic)
                print "pred_label[0].shape:", pred_label[0].shape
                # print pred_label[0]

                # Evalute it: Only the first image, the batchsize should be set to 1
                self.feed_batch_label(image_file_list, cursor, cursor_to)
                print "gt_label[0].shape:", self.batch_label[0].shape

                # Save result
                if self.save_result_or_not:
                    self.save_result(image_file_list, cursor, cursor_to, pred_label[0])
                logging.info("Speed: %.2f images/sec" % (self.batch_size * 1.0 / (time.time() - tic)))

                # ret = self.eval_predict_gt(pred_label, self.batch_label)
                ret = self.eval_predict_gt(pred_label[0], self.batch_label[0])
                self.count.value += 1
                # Update cursor
                cursor += self.batch_size
            except Exception as e:
                logging.info("Error happen: {}".format(e))
                cursor += self.batch_size
        # out_f.close()
        self.save_model_desc()
        print "total pure time consuming :{}s".format(total_pure_time * 1000)

    def read_image(self, img_file):
        img = Image.open(img_file)
        w, h = img.size
        if self.use_half_image:
            w = w // 2
            h = h // 2
            img = img.resize((w, h))
        if (h, w) != self.batch_data_shape[2:]:
            img = img.resize((self.batch_data_shape[3], self.batch_data_shape[2]))
        img = np.array(img).transpose(2, 0, 1)
        if img.shape != self.batch_data_shape[1:]:
            logging.info("Shape not match: {} vs. {}".format(img.shape, self.batch_data_shape[1:]))
            return None
        return img

    def read_label(self, gt_label_file):
        img = Image.open(gt_label_file)
        if self.use_half_image:
            w, h = img.size
            w = w // 2
            h = h // 2
            img = img.resize((w, h), Image.NEAREST)
        if self.upsample_or_not:
            if (h, w) != self.batch_data_shape[2:]:
                img = img.resize((self.batch_data_shape[3], self.batch_data_shape[2]), Image.NEAREST)
        else:
            img = img.resize((self.batch_label_shape[2], self.batch_label_shape[1]), Image.NEAREST)
        return np.array(utils_2d.KdSegId(self.label_map_data)(img))

    def feed_batch_data(self, image_file_list, cursor, cursor_to):
        flag = True
        for i in range(cursor_to - cursor):
            img_file = image_file_list[cursor + i]
            logging.info("Processing file: %s" % (img_file))
            img_info = self.read_image(img_file)
            if img_info is not None:
                self.batch_data[0][i] = img_info
            else:
                flag = False
        return flag

    def feed_batch_label(self, image_file_list, cursor, cursor_to):
        for i in range(cursor_to - cursor):
            image_file = image_file_list[cursor + i]
            gt_label_file = image_file[:-3] + "png"
            assert os.path.exists(gt_label_file), "File not exists: {}".format(gt_label_file)
            self.batch_label[0][i] = self.read_label(gt_label_file)

    def save_result(self, image_file_list, cursor, cursor_to, pred_label):
        pred_label = pred_label.astype(np.uint8)
        for i in range(cursor_to - cursor):
            save_img = Image.fromarray(utils_2d.KdSegId2Color(self.label_id_to_name)(pred_label[i]))
            image_file = image_file_list[cursor + i]
            label_file = os.path.join(self.output_dir, os.path.basename(image_file)[:-3] + "png")

            save_img.save(label_file)

    def eval_predict_gt(self, pred, gt_label):

        assert len(pred) == len(gt_label)
        self.seg_all_metric.update(gt_label, pred)
        self.seg_all_metric.get()
        self.seg_metric.reset()
        self.seg_metric.update(gt_label, pred)
        names, values = self.seg_metric.get()
        for name, value in zip(names, values):
            if "accs" == name:
                accs = value
            if "ious" == name:
                ious = value
        # iou info: (cls_id, iou, pred_cls_pixel_num, gt_label_cls_pixle_num)
        iou_info_array = []
        for idx, iou in enumerate(ious):
            if np.isnan(iou):
                continue
            pred_cls_pixel_num = np.sum(pred[0].astype(np.int) == idx)
            gt_label_cls_pixle_num = np.sum(gt_label[0].astype(np.int) == idx)
            iou_info_array.append((idx, iou, pred_cls_pixel_num, gt_label_cls_pixle_num))
        return self.check_false(iou_info_array)

    def check_false(self, iou_info_array):
        # sort based on iou
        iou_info_array = sorted(iou_info_array, key=lambda d: d[1])
        logging.info(iou_info_array)
        for iou_info in iou_info_array:
            max_pixel_num = max(iou_info[2:])
            if iou_info[1] < self.iou_thresh_low and max_pixel_num >= self.min_pixel_num:
                return iou_info
        return None

    def save_model_desc(self):
        # Save model description into file

        names, values = self.seg_all_metric.get()
        assert len(self.label_id_to_name) == len(values[4]) == len(values[5] == len(values[6])), "False"
        model_desc_file = self.model_desc_file
        with open(model_desc_file, "w") as f:
            f.write("Overall\n")
            f.write("\t".join(names[0:4]) + "\n")
            f.write("\t".join(map(str, values[0:4])) + "\n")
            print "\n**** Final Result: \t".join(map(str, values[0:4]))
            f.write("Details\n")
            f.write("\t".join(["id", "sub-class", "recall", "accuracy", "iou"]) + "\n")
            for idx in xrange(len(self.label_id_to_name)):
                assert str(idx) in self.label_id_to_name
                f.write("\t".join([
                    str(idx), self.label_id_to_name[str(idx)][0].encode("utf8"),
                    str(values[4][idx]),
                    str(values[5][idx]),
                    str(values[6][idx])
                ]))
                f.write("\n")


if __name__ == '__main__':
    import json
    sour_dir = "/data/deeplearning/train_platform/trt_eval_sour/s_data"
    output_dir = "/data/deeplearning/train_platform/trt_eval_sour/c_data"
    weights_file = "/data/deeplearning/train_platform/train_task/gd_full/output/models/kddata_cls17_gd_full_ep-0300.params.h1024w1224.plan"
    json_path = os.path.dirname(weights_file)
    json_path = os.path.join(json_path, 'map_label.json')
    with open(json_path, 'r') as f:
        datas = f.read()
    seg_label_li = json.loads(datas)
    input_size = (1, 3, 1024, 1224)
    output_size = (1, 17, 1024, 1224)
    eval_data = EvalData(
        sour_dir,
        output_dir,
        weights_file,
        input_size,
        output_size,
        effective_c=0,
        gpu_id=0,
        batch_size=1,
        height=1024,
        width=1224,
        use_half_image=False,
        save_result_or_not=True,
        iou_thresh_low=0.5,
        min_pixel_num=2000,
        flip=False,
        upsample_or_not=False,
        seg_label_li=seg_label_li)

    eval_data.run()
