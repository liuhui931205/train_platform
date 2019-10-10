#!/usr/bin/env python
# encoding=utf-8

import sys

sys.path.insert(0, "/opt/kd-seg.template/segmentation")
import os
import time
import copy
import logging
import mxnet as mx
import numpy as np
import shutil
from PIL import Image
from Apps.libs.s_eva_info import utils_2d
from Apps.libs.s_eva_info import val_metric_v2
from Apps.libs.s_eva_info import symbol

symbol.cfg['workspace'] = 1024
symbol.cfg['bn_use_global_stats'] = True


class EvalData(object):

    def __init__(self,
                 weights_file,
                 record_list_file,
                 output_dir,
                 model_desc_file,
                 gpu_id,
                 num_classes,
                 image_type,
                 use_half_image=False,
                 save_result_or_not=False,
                 iou_thresh_low=0.5,
                 min_pixel_num=2000,
                 flip=False,
                 label_map_data=None,
                 label_data=None):
        self.weights_file = weights_file
        self.record_list_file = record_list_file
        self.output_dir = output_dir
        self.model_desc_file = model_desc_file
        self.label_map_data = label_map_data
        self.label_data = label_data
        self.use_half_image = use_half_image
        self.save_result_or_not = save_result_or_not

        # Thresholds
        self.iou_thresh_low = iou_thresh_low
        self.min_pixel_num = min_pixel_num

        # device
        gpus = [gpu_id]
        context = [mx.gpu(gpu_id) for gpu_id in gpus]
        network, net_args, net_auxs = self.load_weights(self.weights_file)

        # Set batch size
        self.batch_size = 1

        # number of classes
        self.num_classes = num_classes

        self.flip = flip
        if image_type != 'full':
            self.h = 1024
            self.w = 2448
        else:
            self.h = 2048
            self.w = 2448

        if self.use_half_image:
            self.h = self.h // 2
            self.w = self.w // 2

        # Module
        self.batch_data_shape = (self.batch_size, 3, self.h, self.w)
        provide_data = [("data", self.batch_data_shape)]
        batch_label_shape = (self.batch_size, self.h, self.w)
        provide_label = [("softmax_label", batch_label_shape)]
        self.mod = mx.mod.Module(network, context=context)
        self.mod.bind(provide_data, provide_label, for_training=False, force_rebind=True)
        self.mod.init_params(arg_params=net_args, aux_params=net_auxs)

        # Upsampling module
        input_data = mx.symbol.Variable(name='data')
        # upsampling = mx.symbol.UpSampling(input_data,
        #                                  scale=1,
        #                                  num_filter=self.num_classes,
        #                                  sample_type='bilinear',
        #                                  name="upsampling_to_input_size",
        #                                  workspace=512)
        argmax_sym = mx.sym.argmax(data=input_data, axis=1)
        self.upsample_mod = mx.mod.Module(argmax_sym, context=context, data_names=['data'], label_names=[])
        self.upsample_mod.bind(
            data_shapes=[('data', (self.batch_size, self.num_classes, self.h, self.w))],
            label_shapes=None,
            for_training=False,
            force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)

        # batch data & batch label
        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]

        # metric
        self.seg_all_metric = val_metric_v2.SegMetric()
        self.seg_all_metric.set_nclass(self.num_classes)
        self.seg_all_metric.reset()

        self.seg_metric = val_metric_v2.SegMetric()
        self.seg_metric.set_nclass(self.num_classes)
        self.seg_metric.reset()

        logging.info("num_classes: {}".format(num_classes))

    def __del__(self):
        pass

    def run(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.predict_dir = os.path.join(self.output_dir, "predict")
        if not os.path.exists(self.predict_dir):
            os.makedirs(self.predict_dir)

        self.diff_dir = os.path.join(self.output_dir, "diff")
        if not os.path.exists(self.diff_dir):
            os.makedirs(self.diff_dir)

        self.doubt_dir = os.path.join(self.output_dir, "doubt")
        if not os.path.exists(self.doubt_dir):
            os.makedirs(self.doubt_dir)

        doubt_file = os.path.join(self.output_dir, "doubt.list")
        out_f = open(doubt_file, "w")

        image_file_list = []
        for line in file(self.record_list_file):
            line = line.strip().split("\t")
            image_file_list.append(line[2])
        assert len(image_file_list) != 0
        logging.info("number_of_images: %d" % (len(image_file_list)))

        cursor = 0
        number_of_samplers = len(image_file_list)
        while cursor < number_of_samplers:
            try:
                cursor_to = min(number_of_samplers, cursor + self.batch_size)
                if not self.feed_batch_data(image_file_list, cursor, cursor_to):
                    cursor += self.batch_size
                    continue

                # Predicting
                tic = time.time()
                pred_label = self.predict()

                # Evalute it: Only the first image, the batchsize should be set to 1
                self.feed_batch_label(image_file_list, cursor, cursor_to)

                # Save result
                if self.save_result_or_not:
                    self.save_result(image_file_list, cursor, cursor_to, pred_label[0])
                logging.info("Speed: %.2f images/sec" % (self.batch_size * 1.0 / (time.time() - tic)))

                ret = self.eval_predict_gt(pred_label, self.batch_label)
                if ret is not None:
                    class_index, min_iou, pred_cls_pixel_num, gt_label_cls_pixle_num = ret
                    image_file = image_file_list[cursor]
                    label_file = os.path.join(
                        os.path.dirname(image_file), "label-" + os.path.basename(image_file)[:-3] + "png")
                    logging.info("{}\t{}\t{}".format(image_file, class_index, min_iou))

                    # File list
                    out_f.write("{}\t{}\t{}\t{}\t{}".format(image_file, class_index, min_iou, pred_cls_pixel_num,
                                                            gt_label_cls_pixle_num))
                    out_f.write("\n")
                    # Copy file
                    assert os.path.exists(image_file)
                    assert os.path.exists(label_file)
                    shutil.copy(image_file, self.doubt_dir)
                    shutil.copy(label_file, self.doubt_dir)
                # Update cursor
                cursor += self.batch_size
            except Exception as e:
                logging.info("Error happen: {}".format(e))
                cursor += self.batch_size
        out_f.close()
        self.save_model_desc()

    def load_weights(self, weights_file):
        assert os.path.exists(weights_file)
        prefix = weights_file.split("_ep-")[0] + "_ep"
        epoch = int(weights_file.split("_ep-")[1].split(".")[0])
        logging.info("prefix: {}, epoch: {}".format(prefix, epoch))
        network, net_args, net_auxs = mx.model.load_checkpoint(prefix, epoch)
        return network, net_args, net_auxs

    def read_image(self, img_file):
        img = Image.open(img_file)
        if self.use_half_image:
            w, h = img.size
            w = w // 2
            h = h // 2
            img = img.resize((w, h))
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

    def gpu_upsampling(self, data):
        self.upsample_mod.forward(mx.io.DataBatch(data=data), is_train=False)
        outputs = self.upsample_mod.get_outputs()
        return outputs

    def predict(self):
        self.mod.forward(mx.io.DataBatch(data=self.batch_data, label=self.batch_label), is_train=False)
        preds = copy.deepcopy(self.mod.get_outputs())
        if self.flip:
            flip_batch_data = []
            for d in self.batch_data:
                flip_batch_data.append(mx.nd.array(d.asnumpy()[:, :, :, ::-1]))
            self.mod.forward(mx.io.DataBatch(data=flip_batch_data, label=self.batch_label), is_train=False)
            flip_preds = self.mod.get_outputs()
            merge_preds = []
            for i, pred in enumerate(preds):
                # change left-lane and right-lane dimension when flipplig
                flipped_pred = flip_preds[i].asnumpy()[:, :, :, ::-1]
                flipped_pred[:, [1, 2], :, :] = flipped_pred[:, [2, 1], :, :]
                merge_preds.append(mx.nd.array((0.5 * pred.asnumpy() + \
                                                0.5 * flipped_pred)))
            preds = merge_preds
        pred_label = self.gpu_upsampling(preds)
        return pred_label

    def save_result(self, image_file_list, cursor, cursor_to, pred_label):
        pred_label = pred_label.asnumpy().astype(np.uint8)
        for i in range(cursor_to - cursor):
            save_img = Image.fromarray(utils_2d.KdSegId2Color(self.label_data)(pred_label[i]))
            image_file = image_file_list[cursor + i]
            label_file = os.path.join(self.predict_dir, "label-" + os.path.basename(image_file)[:-3] + "png")
            save_img.save(label_file)
            predict_file = os.path.join(self.predict_dir, os.path.basename(image_file)[:-3] + "png")
            Image.fromarray(pred_label[i]).save(predict_file)

            # Save diff image
            gt_label = self.batch_label[0][i].asnumpy()
            invalid_mask = np.logical_not(np.in1d(gt_label, [255], invert=True)).reshape(gt_label.shape)
            diff_file = os.path.join(self.diff_dir, os.path.basename(image_file)[:-3] + "png")
            Image.fromarray((invalid_mask * 255 + (gt_label == pred_label[i]) * 127).astype(np.uint8)) \
                .save(diff_file)

    def eval_predict_gt(self, pred, gt_label):
        assert self.batch_size == 1, "Single batch only."
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
            pred_cls_pixel_num = np.sum(pred[0].asnumpy().astype(np.int) == idx)
            gt_label_cls_pixle_num = np.sum(gt_label[0].asnumpy().astype(np.int) == idx)
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
        label_id_to_name = {}
        # for k, v in self.label_data.items():
        #     if int(k) == 255 or int(k) == 254:
        #         continue
        #     label_id_to_name[k] = v
        for kd_label in self.label_data:
            label_id = kd_label['categoryId']
            if int(label_id) == 255 or int(label_id) == 254:
                continue
            if not label_id in label_id_to_name:
                label_id_to_name[label_id] = (kd_label['name'], kd_label['id'], kd_label['color'])

        names, values = self.seg_all_metric.get()
        assert len(label_id_to_name) == len(values[4]) == len(values[5] == len(values[6])), "False"
        model_desc_file = self.model_desc_file
        with open(model_desc_file, "w") as f:
            f.write("Overall\n")
            f.write("\t".join(names[0:4]) + "\n")
            f.write("\t".join(map(str, values[0:4])) + "\n")
            f.write("Details\n")
            f.write("\t".join(["id", "sub-class", "recall", "accuracy", "iou"]) + "\n")
            for idx in xrange(len(label_id_to_name)):
                assert str(idx) in label_id_to_name
                f.write("\t".join([
                    str(idx), label_id_to_name[str(idx)][0].encode("utf8"),
                    str(values[4][idx]),
                    str(values[5][idx]),
                    str(values[6][idx])
                ]))
                f.write("\n")

        cmp_txt_file = os.path.dirname(model_desc_file) + "/cmp.txt"
        with open(cmp_txt_file, "w") as f:
            f.write(str(values[0]) + "\n")
            f.write(str(values[1]) + "\n")
            f.write(str(values[2]) + "\n")
            f.write(str(values[3]) + "\n")
            for v in values[4]:
                f.write(str(v) + "\n")
            for v in values[5]:
                f.write(str(v) + "\n")
            for v in values[6]:
                f.write(str(v) + "\n")
        print("---------------eval_info finished------------------")


# def parse_args():
#     parser = argparse.ArgumentParser(description='Evaluate a network')
#     # weights_file, record_list_file, output_dir
#     parser.add_argument('--weights-file', dest='weights_file', help='',
#                         default="./output/models/model.params", type=str)
#     parser.add_argument('--record-list-file', dest='record_list_file', help='',
#                         default="./data/file.list", type=str)
#     parser.add_argument('--output-dir', dest='output_dir', help='',
#                         default="./output/eval", type=str)
#     parser.add_argument('--model-desc-file', dest='model_desc_file', help='',
#                         default="./output/eval/model.desc", type=str)
#     parser.add_argument('--use-half-image', dest='use_half_image', action='store_true', help='')
#     parser.add_argument('--save-result-or-not', dest='save_result_or_not', action='store_true', help='')
#     parser.add_argument('--iou-thresh-low', dest='iou_thresh_low', help='', default=0.1, type=float)
#     parser.add_argument('--min-pixel-num', dest='min_pixel_num', help='', default=2000, type=int)
#     parser.add_argument('--gpu-id', dest='gpu_id', help='', default=7, type=int)
#     parser.add_argument('--num-classes', dest='num_classes', help='', default=13, type=int)
#     parser.add_argument('--flip', dest='flip', action='store_true', help='')
#     args = parser.parse_args()
#     return args

if __name__ == '__main__':
    # args = parse_args()
    log_fmt = '%(levelname)s: %(asctime)s: %(filename)s:%(lineno)d * %(thread)d %(message)s'
    logging.basicConfig(format=log_fmt, datefmt='%Y%m%d-%H:%M:%S', level=logging.INFO)
    # params = {
    #     "use_half_image": args.use_half_image,
    #     "save_result_or_not": args.save_result_or_not,
    #     "iou_thresh_low": args.iou_thresh_low,
    #     "min_pixel_num": args.min_pixel_num,
    #     "gpu_id": args.gpu_id,
    #     "num_classes": args.num_classes,
    #     "flip": args.flip
    # }
    params = {
        "use_half_image": True,
        "save_result_or_not": False,
        "iou_thresh_low": 0.1,
        "min_pixel_num": 2000,
        "gpu_id": (8, 9),
        "num_classes": 13,
        "flip": True
    }
    weights_file = "/data/deeplearning/train_platform/train_task/shenzhen/output/models/kddata_cls13_shenzhen_ep-0300.params"
    record_list_file = "/data/deeplearning/train_platform/data/shenzhen20180903/kd_all_test.lst"
    output_dir = "/data/deeplearning/train_platform/train_task/shenzhen/output/tracker"
    model_desc_file = "/data/deeplearning/train_platform/train_task/shenzhen/output/tracker/model.desc"
    eval_data = EvalData(weights_file, record_list_file, output_dir, model_desc_file, **params)
    eval_data.run()
