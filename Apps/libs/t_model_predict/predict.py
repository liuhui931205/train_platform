# -*-coding:utf-8-*-

import sys
import os
sys.path.insert(0, "/opt/liuhui/train_platform/Apps/libs/t_model_predict")
from Apps.libs.t_model_predict.rcnn.symbol import *
from Apps.libs.t_model_predict.rcnn.utils.load_model import load_param
from Apps.libs.t_model_predict.rcnn.core.module import MutableModule
from Apps.libs.t_model_predict.rcnn.processing.bbox_transform import nonlinear_pred, clip_boxes
from Apps.libs.t_model_predict.rcnn.processing.nms import py_nms_wrapper
from Apps.libs.t_model_predict.rcnn.io.image import transform
from Apps.libs.t_model_predict.rcnn.config import *
# from labels import object_labels
# from labels import object_labels_sign as object_labels
from Apps.utils.utils import object_labels_sign
import numpy as np
import copy
import cv2
import time
import random
from PIL import Image
from flask import current_app

bbox_pred = nonlinear_pred


class ModelMaskrcnn(object):

    class OneDataBatch():

        def __init__(self, img_ori):
            # transform the img into: (1, 3, H, W) (RGB order)
            img = transform(img_ori, config.PIXEL_MEANS)
            im_info = mx.nd.array([[img.shape[2], img.shape[3], 1.0]])
            self.data = [mx.nd.array(img), im_info]
            self.label = None
            self.provide_label = None
            self.provide_data = [("data", (1, 3, img.shape[2], img.shape[3])), ("im_info", (1, 3))]

    def __init__(self, gpu_id=0, model_path=None, seg_label_li=None):
        """
          param mode:
              "instace" : 1000 * (cls_id) + instance_id
              "class"   : cls_id
              "pixel"   : 0/1
        """
        generate_config("resnet_fpn", "KuandengSign")
        ctx = [mx.gpu(gpu_id)]

        # model path
        self.model_path = model_path
        self.thresh = 0.6
        self.nms_thresh = 0.2
        self.use_global_nms = True
        if seg_label_li:
            self.seg_label_li = seg_label_li
        else:
            self.seg_label_li = object_labels_sign
        self.NUM_ANCHORS = config.NUM_ANCHORS
        # self.NUM_CLASSES = config.NUM_CLASSES
        self.NUM_CLASSES = len(self.seg_label_li)
        # self.CLASSES = ('__background__', 'v', 'p', 's', 'l')
        # self.class_id = [0, 1, 2, 3, 4]

        self.CLASSES = ('__background__', 's', 'l')
        self.class_id = [0, 1, 2]

        sym = get_resnet_fpn_mask_test(num_classes=self.NUM_CLASSES, num_anchors=self.NUM_ANCHORS)

        # Load model
        arg_params, aux_params = load_param(self.model_path, convert=False, ctx=ctx, process=True)

        # 2048 * 0.625, 2560 * 0.625
        # config.SCALES
        max_image_shape = (1, 3, 1280, 1600)
        max_data_shapes = [("data", max_image_shape), ("im_info", (1, 3))]

        self.height = max_image_shape[2]
        self.width = max_image_shape[3]

        # self.mod = mx.mod.Module(symbol=sym, context=ctx, data_names=["data", "im_info"])
        self.mod = MutableModule(
            symbol=sym, data_names=["data", "im_info"], label_names=None, max_data_shapes=max_data_shapes, context=ctx)
        self.mod.bind(data_shapes=max_data_shapes, label_shapes=None, for_training=False, force_rebind=True)
        self.mod.init_params(arg_params=arg_params, aux_params=aux_params)

        # nms tool
        self.nms = py_nms_wrapper(self.nms_thresh)

        # color info
        self.class_colors = [None] * self.NUM_CLASSES
        # for label in object_labels:
        for label in self.seg_label_li:
            if not label["trainId"] in self.class_id:
                continue
            self.class_colors[label["trainId"]] = (label["color"][2], label["color"][1], label["color"][0])

    def check_valid(self, bbox):
        if bbox[2] == bbox[0] or bbox[3] == bbox[1] or bbox[0] == bbox[1] or bbox[2] == bbox[3]:
            return False
        # The box is on the floor
        if bbox[3] >= self.height - 1 or bbox[1] >= self.height - 1:
            return False
        return True

    def global_nms_bbox(self, all_boxes, all_masks):
        expand_all_boxes = []
        for cls in self.seg_label_li:
            if "__background__" == cls["categoryId"]:
                continue
            # cls_ind = self.CLASSES.index(cls)
            cls_ind = cls["trainId"]
            cls_boxes = all_boxes[cls_ind]
            expand_all_boxes.append(np.hstack((cls_boxes, np.tile(cls_ind, (cls_boxes.shape[0], 1)).astype(np.int))))
        all_boxes_set = np.concatenate(expand_all_boxes, axis=0)
        all_masks_set = np.concatenate(all_masks[1:], axis=0)
        all_keep = self.nms(all_boxes_set[:, :-1])
        all_keep_boxes = all_boxes_set[all_keep, :].astype(np.float32)
        all_keep_masks = all_masks_set[all_keep, :].astype(np.float32)

        for cls in self.seg_label_li:
            if "__background__" == cls["categoryId"]:
                continue
            cls_ind = cls["trainId"]
            keep = np.where(all_keep_boxes[:, -1] == cls_ind)[0]
            all_boxes[cls_ind] = all_keep_boxes[keep, :-1]
            all_masks[cls_ind] = all_keep_masks[keep]

    def do(self, image_data, dest_file=None):
        pred_data = None
        try:
            image = np.asarray(bytearray(image_data), dtype="uint8")
            im = cv2.imdecode(image, cv2.IMREAD_COLOR)
            raw_image = copy.copy(im)
            # print(raw_image.shape)

            # enlarge the original image and resize
            scale_ind = 0
            target_size = config.SCALES[scale_ind][0]
            max_size = config.SCALES[scale_ind][1]
            # pad img to fixed size
            h, w = im.shape[:2]
            h_scale_ratio = h * 1.0 / target_size
            w_scale_ratio = w * 1.0 / max_size
            if h_scale_ratio > w_scale_ratio:
                # pad x
                w = int(max_size * h_scale_ratio)
            else:
                # pad y
                h = int(target_size * w_scale_ratio)
            if h != im.shape[0] or w != im.shape[1]:
                target_im = np.zeros((h, w, im.shape[2]), dtype=np.uint8)
                target_im[:im.shape[0], :im.shape[1], :] = im
                im = target_im

            # resize im: * 0.75
            # 0.625
            im_scale = 0.625
            im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)

            # make data batch
            batch = self.OneDataBatch(im)

            self.mod.forward(batch, False)
            results = self.mod.get_outputs()
            mx.nd.waitall()
            # print("get result")
            output = dict(zip(self.mod.output_names, results))
            rois = output['rois_output'].asnumpy()[:, 1:]
            scores = output['cls_prob_reshape_output'].asnumpy()[0]
            bbox_deltas = output['bbox_pred_reshape_output'].asnumpy()[0]
            mask_output = output['mask_prob_output'].asnumpy()

            pred_boxes = bbox_pred(rois, bbox_deltas)
            # may be should change
            pred_boxes = clip_boxes(pred_boxes, [im.shape[0], 2448 // 4 * 3])
            boxes = pred_boxes

            all_boxes = [None for _ in xrange(len(self.seg_label_li))]
            all_masks = [None for _ in xrange(len(self.seg_label_li))]
            label = np.argmax(scores, axis=1)
            label = label[:, np.newaxis]

            for cls in self.seg_label_li:
                cls_ind = cls["trainId"]
                cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
                # print(cls_boxes.shape)
                cls_masks = mask_output[:, cls_ind, :, :]
                cls_scores = scores[:, cls_ind, np.newaxis]
                keep = np.where((cls_scores >= self.thresh) & (label == cls_ind))[0]
                cls_masks = cls_masks[keep, :, :]
                dets = np.hstack((cls_boxes, cls_scores)).astype(np.float32)[keep, :]
                keep = self.nms(dets)
                # print("keep:", keep)
                all_boxes[cls_ind] = dets[keep, :]
                all_masks[cls_ind] = cls_masks[keep, :, :]

            # apply global nms
            if self.use_global_nms:
                self.global_nms_bbox(all_boxes, all_masks)

            boxes_this_image = [[]] + [all_boxes[j] for j in range(1, len(self.seg_label_li))]
            masks_this_image = [[]] + [all_masks[j] for j in range(1, len(self.seg_label_li))]

            self.draw_detection_mask(raw_image, boxes_this_image, masks_this_image, 1.0 / im_scale, dest_file)
        except Exception as e:
            current_app.logger.error("recognition error:{}".format(repr(e)))
        finally:
            return pred_data

    def make_result_img(self, img_ori, boxes_this_image, masks_this_image, dest, scale=1.0):
        im = copy.copy(img_ori)
        h, w, _ = im.shape
        instance_id_img = np.zeros((h, w)).astype(np.int32)
        # backgroundId = 0
        # instanceImg = Image.new("I", (w, h), backgroundId)
        # drawer = ImageDraw.Draw(instanceImg)
        # instance_id: (bbox, score)
        instance_bbox = {}
        for j, name in enumerate(self.seg_label_li):
            if name == '__background__':
                continue
            dets = boxes_this_image[j]
            masks = masks_this_image[j]
            for i in range(len(dets)):
                bbox = dets[i, :4] * scale
                if not self.check_valid(bbox):
                    continue
                instance_id = j * 1000 + (i + 1)
                score = dets[i, -1]
                bbox = map(int, bbox)
                mask = masks[i, :, :]
                mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                mask[mask > 0.5] = instance_id
                mask[mask <= 0.5] = 0
                instance_id_img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = mask.astype(np.int32)
                instance_bbox[instance_id] = (bbox, score)

                # cv2.fillPoly(instance_id_img,mask.astype(np.int32), instance_id)

        blank_image = Image.fromarray(instance_id_img, mode="I")
        # pred_data = StringIO.BytesIO()
        # pred_data = im[1].tostring()
        blank_image.save(dest, format="PNG")

        # if dest is not None:
        #     with open(dest, "wb") as f:
        #         f.write(pred_data)
        return instance_bbox, instance_id_img

    def draw_detection_mask(self, im_array, boxes_this_image, masks_this_image, scale, filename):
        class_names = self.seg_label_li
        color_white = (255, 255, 255)
        im = im_array
        # change to bgr
        #im = cv2.cvtColor(im, cv2.cv.CV_RGB2BGR)
        for j, name in enumerate(class_names):
            if name["categoryId"] == '__background__':
                continue
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0,
                                                                                    256))    # generate a random color
            dets = boxes_this_image[j]
            masks = masks_this_image[j]
            for i in range(len(dets)):
                bbox = dets[i, :4] * scale
                score = dets[i, -1]
                bbox = map(int, bbox)
                cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color=color, thickness=2)
                cv2.putText(
                    im,
                    '%s %.3f' % (name["categoryId"], score), (bbox[0], bbox[1] + 10),
                    color=color_white,
                    fontFace=cv2.FONT_HERSHEY_COMPLEX,
                    fontScale=0.5)
                mask = masks[i, :, :]
                mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
                mask[mask > 0.5] = 1
                mask[mask <= 0.5] = 0
                mask_color = random.randint(0, 255)
                c = random.randint(0, 2)
                target = im[bbox[1]:bbox[3], bbox[0]:bbox[2], c] + mask_color * mask
                target[target >= 255] = 255
                im[bbox[1]:bbox[3], bbox[0]:bbox[2], c] = target
        cv2.imwrite(filename, im)


class Task:

    def __init__(self, img_data, dest_path, exit_flag=False):
        self.img_data = img_data
        self.dest_path = dest_path
        self.exit_flag = exit_flag


def do_seg(gpu_id, task_queue, model_path, count, seg_label_li=None, lock=None):
    net = ModelMaskrcnn(gpu_id=gpu_id, model_path=model_path)
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
            lock.acquire()
            net.do(image_data=img_data, dest_file=dest_path)

            count.value += 1
            end = time.time()
            lock.release()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--dir', type=str, required=True)
    # parser.add_argument('--gpu_id', type=int, default=0)
    #
    # args = parser.parse_args()
    seg_label_li = [{
        "className": "background",
        "categoryId": "__background__",
        "trainId": 0,
        "color": (0, 0, 0)
    }, {
        "className": "traffic_signs",
        "categoryId": "s",
        "trainId": 1,
        "color": (0, 0, 255)
    }, {
        "className": "traffic_lights",
        "categoryId": "l",
        "trainId": 2,
        "color": (255, 0, 255)
    }]
    print mx.__path__
    image_dir = "/opt/test"
    # image_dir = args.dir
    # model_net = ModelMaskrcnn(gpu_id=args.gpu_id)
    # model_net = ModelMaskrcnn(gpu_id=9)
    model_net = ModelMaskrcnn(gpu_id=9, model_path="/opt/test/final-0000.params", seg_label_li=seg_label_li)
    proc_list = []
    file_list = os.listdir(image_dir)
    for id_ in file_list:
        if not id_.endswith("jpg"):
            continue
        proc_list.append(id_)

    for id_ in proc_list:
        file_path = os.path.join(image_dir, id_)
        dest_dir = os.path.join(image_dir, "mask_rcnn")
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        name_list = str(id_).split(".")
        name_only = name_list[0]

        file_id = name_only + ".png"
        dest_path = os.path.join(dest_dir, file_id)

        try:
            start = time.time()
            with open(file_path, "rb") as f:
                img = f.read()
                model_net.do(image_data=img, dest_file=dest_path)
            end = time.time()
            print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))


if __name__ == '__main__':
    main()
