#!/usr/bin/env python
# encoding=utf-8

import sys

sys.path.insert(0, "/opt/kd-seg.template/segmentation")
import os
import time
import copy
import cv2
import mxnet as mx
import numpy as np
from PIL import Image
import utils_2d
import val_metric_v2
import symbol
import json
from flask import current_app

os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"
symbol.cfg['workspace'] = 1024
symbol.cfg['bn_use_global_stats'] = True


class EvalData(object):

    def __init__(self,
                 weights_file,
                 gpu_id,
                 num_classes,
                 image_type,
                 output_confidence,
                 info,
                 use_half_image=False,
                 save_result_or_not=False,
                 iou_thresh_low=0.5,
                 min_pixel_num=2000,
                 flip=False,
                 label_map_data=None,
                 label_data=None):
        self.weights_file = weights_file
        self.label_map_data = label_map_data
        self.label_data = label_data
        self.use_half_image = use_half_image
        self.save_result_or_not = save_result_or_not
        self.info = info
        # Thresholds
        self.iou_thresh_low = iou_thresh_low
        self.min_pixel_num = min_pixel_num
        self.image_type = image_type

        # device
        gpus = [gpu_id]
        context = [mx.gpu(gpu_id) for gpu_id in gpus]
        network, net_args, net_auxs = self.load_weights(self.weights_file)

        # Set batch size
        self.batch_size = 1
        self.upsample_or_no = True
        self.label_to_color = np.zeros((256, 3), dtype=np.uint8)
        for label in self.label_data:
            if not int(label["categoryId"]) in range(0, 256):
                continue
            self.label_to_color[int(label["categoryId"])] = (label["color"][2], label["color"][1], label["color"][0])
            if label["color"] == [128, 64, 128]:
                self.road_id = int(label["categoryId"])
            elif label["color"] == [64, 64, 32]:
                self.other_id = int(label["categoryId"])

        # number of classes
        self.num_classes = num_classes

        self.flip = flip
        if self.image_type != 'full':
            self.h = 1024
            self.w = 2448
            self.result_shape = [1024, 2448]
        else:
            self.h = 2048
            self.w = 2448
            self.result_shape = [2048, 2448]

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
        # input_data = mx.symbol.Variable(name='data')
        self.add_conf = output_confidence
        upsampling_sym, confidence_sym = self.get_upsampling_sym()
        # upsampling = mx.symbol.UpSampling(
        #     input_data,
        #     scale=2,
        #     num_filter=self.num_classes,
        #     sample_type='bilinear',
        #     name="upsampling_to_input_size",
        #     workspace=512)
        # argmax_sym = mx.sym.argmax(data=upsampling, axis=1)
        self.upsample_mod = mx.mod.Module(upsampling_sym, context=context, data_names=['data'], label_names=[])
        if self.add_conf:
            self.confidence_mod = mx.mod.Module(confidence_sym, context=context, data_names=['data'], label_names=[])
        self.upsample_mod.bind(
            data_shapes=[('data', (self.batch_size, self.num_classes, self.h, self.w))],
            label_shapes=None,
            for_training=False,
            force_rebind=True)
        if self.add_conf:
            self.confidence_mod.bind(
                data_shapes=[('data', (1, int(self.num_classes), self.h, self.w))],
                label_shapes=None,
                for_training=False,
                force_rebind=True)
        initializer = mx.init.Bilinear()
        self.upsample_mod.init_params(initializer=initializer)
        if self.add_conf:
            self.confidence_mod.init_params(initializer=initializer)

        # batch data & batch label
        self.batch_data = [mx.nd.empty(info[1]) for info in provide_data]
        # self.batch_label = [mx.nd.empty(info[1]) for info in provide_label]
        batch_label_shapes = (self.batch_size, self.h * 2, self.w * 2)
        provide_labels = [("softmax_label", batch_label_shapes)]
        self.batch_label = [mx.nd.empty(info[1]) for info in provide_labels]

        # metric
        self.seg_metric = val_metric_v2.SegMetric()
        self.seg_metric.set_nclass(self.num_classes)
        self.seg_metric.reset()

        # print("num_classes: {}".format(num_classes))

    def __del__(self):
        pass

    def run(self, flag, img_file, dest_path, update_queue, info_queue):
        img_info = self.read_image(img_file)

        if img_info is not None:
            self.batch_data[0][0] = img_info

        tic = time.time()
        pred_label, confidence_outputs = self.predict()
        if flag:

            label_info = self.read_label(img_file[:-3] + "png")
            if label_info is not None:
                self.batch_label[0][0] = label_info
        # Save result

        # print("Speed: %.2f images/sec" % (self.batch_size * 1.0 / (time.time() - tic)))
            label = self.batch_label[0].asnumpy().ravel()
            pred = pred_label[0].asnumpy().ravel()
            update_task = UpdateInfo(pred_label=pred, batch_label=label)
            update_queue.put(update_task)
            ret = self.eval_predict_gt(pred, label)

        if self.save_result_or_not:
            self.save_result(img_file, dest_path, pred_label[0], confidence_outputs, info_queue)

    def load_weights(self, weights_file):
        assert os.path.exists(weights_file)
        prefix = weights_file.split("_ep-")[0] + "_ep"
        epoch = int(weights_file.split("_ep-")[1].split(".")[0])
        # print("prefix: {}, epoch: {}".format(prefix, epoch))
        network, net_args, net_auxs = mx.model.load_checkpoint(prefix, epoch)
        return network, net_args, net_auxs

    def get_upsampling_sym(self):
        input_data = mx.symbol.Variable(name='data')
        upsampling = mx.symbol.UpSampling(
            input_data,
            scale=2,
            num_filter=int(self.num_classes),
            sample_type='bilinear',
            name="upsampling_preds",
            workspace=512)
        upsampling_sym = mx.sym.argmax(data=upsampling, axis=1)
        confidence_sym = mx.sym.max(data=upsampling, axis=1)
        return upsampling_sym, confidence_sym

    def read_image(self, img_file):
        img = Image.open(img_file)
        with open(img_file,'rb') as f:
            image_data = f.read()
        image = np.asarray(bytearray(image_data), dtype="uint8")
        origin_frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
        origin_frame = origin_frame[:, :, ::-1]
        width = origin_frame.shape[1]
        height = origin_frame.shape[0]
        bottom_half = origin_frame[height - self.result_shape[0]:height, 0:width]
        if self.use_half_image:
            w = width // 2
            if height == 2048 and self.image_type != "full":
                height = height // 2
            h = height // 2
            # img = img.resize((w, h))
            img = np.array(
                Image.fromarray(bottom_half.astype(np.uint8, copy=False)).resize(
                    (w, h), Image.NEAREST))
        img = img.transpose(2, 0, 1)
        if img.shape != self.batch_data_shape[1:]:
            # print("Shape not match: {} vs. {}".format(img.shape, self.batch_data_shape[1:]))
            return None
        return img

    def read_label(self, gt_label_file):
        img = Image.open(gt_label_file)
        if not self.upsample_or_no:
            w, h = img.size
            w = w // 2
            h = h // 2
            img = img.resize((w, h), Image.NEAREST)
        return np.array(utils_2d.KdSegId(self.label_map_data)(img))

    def gpu_upsampling(self, data):
        confidence_outputs = None
        if self.add_conf:
            self.confidence_mod.forward(mx.io.DataBatch(data=data), is_train=False)
            confidence_outputs = self.confidence_mod.get_outputs()
        self.upsample_mod.forward(mx.io.DataBatch(data=data), is_train=False)
        outputs = self.upsample_mod.get_outputs()
        return outputs, confidence_outputs

    def predict(self):
        self.mod.forward(mx.io.DataBatch(data=self.batch_data, label=None), is_train=False)
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
        pred_label, confidence_outputs = self.gpu_upsampling(preds)
        return pred_label, confidence_outputs

    def save_result(self, img_file, dest_path, pred_label, confidence_outputs, info_queue):
        # pred_label = pred_label.asnumpy().astype(np.uint8)
        # save_img = Image.fromarray(utils_2d.KdSegId2Color(self.label_data)(pred_label[0]))
        # save_img.save(dest_path)
        _image = cv2.imread(img_file)
        width = _image.shape[1]
        height = _image.shape[0]
        pred_label = pred_label.asnumpy().squeeze().astype(np.uint8)
        copy_out = copy.deepcopy(pred_label)
        rgb_frame = self.label_to_color[pred_label]
        recog_image = np.zeros((height, width, 3), np.uint8)
        recog_image[height - self.result_shape[0]:height, 0:width] = rgb_frame
        blank_image = np.zeros((height, width, 4), np.uint8)
        blank_image[0:height, 0:width] = (0, 0, 0, 255)

        blank_image[0:height - self.result_shape[0], 0:width] = (0, 0, 0, 0)
        blank_image[0:height, 0:width, 0] = recog_image[:, :, 0]
        blank_image[0:height, 0:width, 1] = recog_image[:, :, 1]
        blank_image[0:height, 0:width, 2] = recog_image[:, :, 2]

        if self.add_conf:
            confidence = (confidence_outputs[0].asnumpy().astype(np.single))
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

            a = np.where((blank_image[:, :, 0] == 128) & (blank_image[:, :, 1] == 64) & (blank_image[:, :, 2] == 128))
            blank_image[a] = [128, 64, 128, 255]

            b = np.where((blank_image[:, :, 0] == 32) & (blank_image[:, :, 1] == 64) & (blank_image[:, :, 2] == 64))
            blank_image[b] = [32, 64, 64, 255]

            whole_con = round(np.mean(copy_con), 4)
            c = np.where(copy_out[:] == self.road_id)
            d = np.where(copy_out[:] == self.other_id)
            do_whole_con = round(
                (np.sum(copy_con) - np.sum(copy_con[c]) - np.sum(copy_con[d])) / (2448 * 1024 - len(c[0]) - len(d[0])),
                4)

            cls = self.num_classes
            s_pixel = np.zeros(cls)
            bins = [i for i in range(cls + 1)]
            (n, _) = np.histogram(pred_label, bins=bins, normed=False)

            s_pixel += n
            o_cls_con = {}
            # cls_con = []
            for i in range(cls):
                c = np.where((copy_out[:] == i))
                # d = np.where((copy_out[:] == i))

                o_cls_con["{}".format(i)] = str(round((np.sum(copy_con[c]) / s_pixel[i]), 4))
                # cls_con.append({i: round((np.sum(confidence[c]) / s_pixel[i]), 4)})
            # model_info = self.weights.split('/')
            dest_name = os.path.basename(dest_path)
            info = self.info(whole_con, do_whole_con, o_cls_con, dest_name[:-11])
            info_queue.put(info)
        pred_data = blank_image

        if dest_path is not None:
            with open(dest_path, "wb") as f:
                img_array = cv2.imencode('.png', pred_data)
                img_data = img_array[1]
                pred_data = img_data.tostring()
                f.write(pred_data)

    def eval_predict_gt(self, pred, gt_label):
        assert self.batch_size == 1, "Single batch only."
        assert len(pred) == len(gt_label)

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
            pred_cls_pixel_num = np.sum(pred.astype(np.int) == idx)
            gt_label_cls_pixle_num = np.sum(gt_label.astype(np.int) == idx)
            iou_info_array.append((idx, iou, pred_cls_pixel_num, gt_label_cls_pixle_num))
        return self.check_false(iou_info_array)

    def check_false(self, iou_info_array):
        # sort based on iou
        iou_info_array = sorted(iou_info_array, key=lambda d: d[1])
        # print(iou_info_array)
        for iou_info in iou_info_array:
            max_pixel_num = max(iou_info[2:])
            if iou_info[1] < self.iou_thresh_low and max_pixel_num >= self.min_pixel_num:
                return iou_info
        return None


class Task:

    def __init__(self, img_name, dest_path, exit_flag=False):
        self.img_name = img_name
        self.dest_path = dest_path
        self.exit_flag = exit_flag


class UpdateInfo:

    def __init__(self, pred_label, batch_label):
        self.pred_label = pred_label
        self.batch_label = batch_label


class Info:

    def __init__(self, origin_w_con, w_con, o_cls_con, trackpointid, exit_flag=False):
        self.or_w_con = origin_w_con
        self.w_con = w_con
        self.o_cls_con = o_cls_con
        # self.cls_con = cls_con
        # self.model = model
        self.trackpointid = trackpointid
        self.exit_flag = exit_flag


def do_seg(flag, image_type, gpu_id, output_confidence, task_queue, model_path, count, cls, label_data, label_map_data,
           lock, update_queue, info_queue):
    info = Info
    net = EvalData(
        model_path,
        gpu_id,
        cls,
        image_type,
        output_confidence,
        info,
        use_half_image=True,
        save_result_or_not=True,
        iou_thresh_low=0.1,
        min_pixel_num=2000,
        flip=False,
        label_map_data=label_map_data,
        label_data=label_data)
    while True:
        if task_queue.empty():
            break

        task = task_queue.get()

        if not isinstance(task, Task):
            break

        if task.exit_flag:
            break

        img_name = task.img_name
        dest_path = task.dest_path

        try:
            start = time.time()
            net.run(flag, img_name, dest_path, update_queue, info_queue)
            lock.acquire()
            count.value += 1
            end = time.time()
            lock.release()
            # print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))


def update(seg_all_metric, update_queue):
    while True:
        if update_queue.empty():
            break
        update_task = update_queue.get()

        if not isinstance(update_task, UpdateInfo):
            break
        pred_label = update_task.pred_label
        batch_label = update_task.batch_label
        seg_all_metric.update(pred_label, batch_label)
        seg_all_metric.get()

    return seg_all_metric


def save_model_desc(seg_all_metric, label_data, model_desc_file):
    # Save model description into file
    label_id_to_name = {}
    # for k, v in self.label_data.items():
    #     if int(k) == 255 or int(k) == 254:
    #         continue
    #     label_id_to_name[k] = v
    for kd_label in label_data:
        label_id = kd_label['categoryId']
        if int(label_id) == 255 or int(label_id) == 254:
            continue
        if not label_id in label_id_to_name:
            label_id_to_name[label_id] = (kd_label['name'], kd_label['id'], kd_label['color'])

    names, values = seg_all_metric.get()
    assert (len(label_id_to_name)) == len(values[4]) == len(values[5] == len(values[6])), "False"
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
    print("-------------------success")

    # cmp_txt_file = os.path.dirname(model_desc_file) + "/cmp.txt"
    # with open(cmp_txt_file, "w") as f:
    #     f.write(str(values[0]) + "\n")
    #     f.write(str(values[1]) + "\n")
    #     f.write(str(values[2]) + "\n")
    #     f.write(str(values[3]) + "\n")
    #     for v in values[4]:
    #         f.write(str(v) + "\n")
    #     for v in values[5]:
    #         f.write(str(v) + "\n")
    #     for v in values[6]:
    #         f.write(str(v) + "\n")
    print("---------------eval_info finished------------------")


if __name__ == '__main__':
    import multiprocessing
    gpus = "6,7,8"
    num_classes = 13
    image_type = "half"
    # eval_data = EvalData(weights_file, record_list_file, output_dir, model_desc_file, **params)
    # eval_data.run()
    task_name = "beijing20190114"
    data_name = "beijing20190114"
    data_type = "all"
    output_confidence = True
    paths = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/models"
    map_path = "/data/deeplearning/train_platform/train_task/" + task_name + "/conf/label_map.txt"
    model_path = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/models/kddata_cls13_beijing_ep-0300.params"
    output_dir = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/tracker"
    model_desc_file = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/models/infoss.txt"
    record_list_file = os.path.join('/data/deeplearning/train_platform/data', data_name,
                                    'kd_' + data_type + '_test.lst')
    gpus = gpus
    gpu_ids = gpus.split(",")
    gpu_ids.sort(reverse=True)
    gpu_count = len(gpu_ids)
    single_count = int(2)
    gpu_threads = []

    seg_all_metric = val_metric_v2.SegMetric()
    seg_all_metric.set_nclass(num_classes)
    seg_all_metric.reset()

    cls = int(13)
    label_map_data = []
    with open(map_path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = (line.strip()).split('\t')
                label_map_data.append(line)
            else:
                label_map_data.pop(0)
                break
    with open(os.path.join(paths, 'map_label.json'), 'r') as d:
        label_data = json.loads(d.read())
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    task_queue = manager.Queue()
    update_queue = manager.Queue()
    count = manager.Value('i', 0)
    for line in file(record_list_file):
        line = line.strip().split("\t")
        name = os.path.basename(line[2])
        name = name[:-3] + "png"
        dest_path = os.path.join("/data/deeplearning/train_platform/train_task/beijing20190114/output/tracker", name)
        task = Task(img_name=line[2], dest_path=dest_path, exit_flag=False)
        task_queue.put(task)

    for i in range(gpu_count * single_count):
        gpu_id = int(gpu_ids[int(i / single_count)])
        t = multiprocessing.Process(
            target=do_seg,
            args=(gpu_id, output_confidence, task_queue, model_path, count, cls, label_data, label_map_data, lock,
                  update_queue))
        gpu_threads.append(t)

    for j in range(single_count):
        for i in range(gpu_count):
            t = gpu_threads[i * single_count + j]
            t.start()
        time.sleep(15)
    for process in gpu_threads:
        process.join()

    seg_all_metric = update(seg_all_metric, update_queue)

    save_model_desc(seg_all_metric, label_data, model_desc_file)
