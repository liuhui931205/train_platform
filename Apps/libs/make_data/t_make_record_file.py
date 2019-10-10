import sys
import os
import json
import numpy as np
import random
from PIL import Image
from PIL import ImageDraw
from Apps.utils.utils import traffic_signs, traffic_lights, pedestrian, vehicles
import argparse
import time


class Task:

    def __init__(self, json_path, exit_flag=False):
        self.json_path = json_path
        self.exit_flag = exit_flag


def do(task_queue):
    t_pro = PreProess()
    while True:
        if task_queue.empty():
            break

        task = task_queue.get()

        if not isinstance(task, Task):
            break

        if task.exit_flag:
            break

        json_path = task.json_path

        try:
            t_pro.make_instance_segmentation_data(json_path)
            # print("Processed {} in {} ms".format(dest_path, str((end - start) * 1000)))
        except Exception as e:
            print(repr(e))


class PreProess(object):

    def __init__(self, dataset_path=None, is_pole=False):
        self.data_path = dataset_path
        # self.root_path = root_path
        self.label_map = {}
        # vehicles = [12, 62, 63, 65, 66, 67, 74]
        # pedestrian = [68]
        # traffic_signs = traffic_signs
        # traffic_lights = traffic_lights
        # for v in vehicles:
        #     self.label_map[v] = 3
        # for v in pedestrian:
        #     self.label_map[v] = 4
        for v in traffic_signs:
            self.label_map[v] = 1
        for v in traffic_lights:
            self.label_map[v] = 2

        # just pole data
        if is_pole:
            poles = [33]
            self.label_map = {}
            for v in poles:
                self.label_map[v] = 1

    def __del__(self):
        pass

    def make_instance_segmentation_data(self, json_path):
        # the data can be processed in parallel
        sub_dir = os.path.dirname(json_path)
        # print "parse file:", json_path
        info = json.load(open(json_path, "r"))
        imgs = info["imgs"]
        for img_key in imgs:
            img_name, suffix = os.path.splitext(img_key)
            # print("img_name:", img_name)
            # load each image info
            # path
            # objects: category polygon bbox id
            image_info = imgs[img_key]
            image_path = os.path.join(sub_dir, image_info['path'])
            img = np.array(Image.open(image_path, 'r'))
            h, w = img.shape[:2]

            backgroundId = 0
            instanceImg = Image.new("I", (w, h), backgroundId)
            # a drawer to draw into the image
            drawer = ImageDraw.Draw(instanceImg)

            for obj in image_info['objects']:
                # print(obj)
                polygon = np.array(obj['polygon']).astype(np.int32)
                #
                category = int(obj['category'])
                if not category in self.label_map:
                    continue
                category = self.label_map[category]
                #
                obj_id = obj['id']
                if obj_id == backgroundId:
                    # ignore background id
                    continue
                bbox = obj['bbox']
                # A category mapping is necessary
                # v : 1000 * c + id
                v = 1000 * category + obj_id
                # print("fill value:", v)
                drawer.polygon(polygon.tolist(), fill=v)
            # save instance image
            ins_img_name = os.path.join(sub_dir, img_name + "-ins.png")
            instanceImg.save(ins_img_name)

    def make_train_val_file_list(self):
        all_files = []
        datas_path = os.path.join(self.data_path, "data")
        dirs = os.listdir(datas_path)
        total = len(dirs)
        for d in dirs:
            sub_dir = os.path.join(datas_path, d)
            if not os.path.isdir(sub_dir):
                continue
            # json_file = os.path.join(sub_dir, "annot_" + d + ".json")
            json_file = os.path.join(sub_dir, d + ".json")
            if not os.path.exists(json_file):
                print "file not exists.", json_file
                continue
            print "parse file:", json_file
            info = json.load(open(json_file, "r"))
            imgs = info["imgs"]
            for img_key in imgs:
                img_name, suffix = os.path.splitext(img_key)
                print("img_name:", img_name)
                image_path = os.path.join(d, img_name + ".jpg")
                ins_seg_path = os.path.join(d, img_name + "-ins.png")
                absolute_image_path = os.path.join(sub_dir, img_name + ".jpg")
                absolute_ins_seg_path = os.path.join(sub_dir, img_name + "-ins.png")
                if not os.path.exists(absolute_image_path):
                    print "file not exists:", absolute_image_path
                    continue
                if not os.path.exists(absolute_ins_seg_path):
                    print "file not exists:", absolute_ins_seg_path
                    continue
                all_files.append((image_path, ins_seg_path))
        # shuffle and save
        random.shuffle(all_files)
        ratio = 0.9
        number_of_train_examples = int(ratio * len(all_files))
        train_files = all_files[0:number_of_train_examples]
        val_files = all_files[number_of_train_examples:]
        # save
        train_list_file_name = os.path.join(self.data_path, "imglists", "train.lst")
        val_list_file_name = os.path.join(self.data_path, "imglists", "val.lst")
        PreProess.save_file_list(train_list_file_name, train_files)
        PreProess.save_file_list(val_list_file_name, val_files)

    def output_bbox_width_and_height(self):
        anchors = []
        dirs = os.listdir(self.data_path)
        for d in dirs:
            sub_dir = os.path.join(self.data_path, d)
            if not os.path.isdir(sub_dir):
                continue
            #json_file = os.path.join(sub_dir, "annot_" + d + ".json")
            json_file = os.path.join(sub_dir, d + ".json")
            if not os.path.exists(json_file):
                continue
            print "parse file:", json_file
            info = json.load(open(json_file, "r"))
            imgs = info["imgs"]
            for img_key in imgs:
                img_name, suffix = os.path.splitext(img_key)
                print("img_name:", img_name)
                # load each image info
                # path
                # objects: category polygon bbox id
                image_info = imgs[img_key]
                for obj in image_info['objects']:
                    bbox = obj['bbox']
                    anchors.append((bbox['xmax'] - bbox['xmin'], bbox['ymax'] - bbox['ymin']))
        anchor_file = os.path.join(self.data_path, "anchors.txt")
        anchor_path = os.path.dirname(anchor_file)
        if not os.path.exists(anchor_path):
            os.makedirs(anchor_path)
        with open(anchor_file, "w") as f:
            for anchor in anchors:
                f.write("%d,%d" % (anchor[0], anchor[1]))
                f.write("\n")

    @staticmethod
    def save_file_list(file_name, file_list):
        dir_path = os.path.dirname(file_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(file_name, "w") as f:
            for item in file_list:
                f.write("%s\t%s" % (item[0], item[1]))
                f.write("\n")


def parse_args():
    parser = argparse.ArgumentParser(description='preprocess selection')
    parser.add_argument('--data', help='', action='store_true')
    parser.add_argument('--list', help='', action='store_true')
    parser.add_argument('--anchor', help='', action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    data_info = {
        "traffic_sign": {
            "dataset_path": "/data/deeplearning/maskrcnn/traffic_sign/released",
            "root_path": "/opt/mx-maskrcnn/data/traffic_sign"
        },
        "shangqi": {
            "dataset_path": "/data/deeplearning/maskrcnn/shangqi/data",
            "root_path": "/opt/mx-maskrcnn/data/shangqi"
        },
        "pole": {
            "dataset_path": "/data/deeplearning/maskrcnn/pole/data",
            "root_path": "/opt/mx-maskrcnn/data/pole"
        },
        "sign": {
            "dataset_path": "/data/deeplearning/maskrcnn/sign/data",
            "root_path": "/opt/mx-maskrcnn/data/sign"
        }
    }
    data_type = "sign"
    dataset_path = data_info[data_type]["dataset_path"]
    root_path = data_info[data_type]["root_path"]
    pre_process = PreProess(dataset_path, root_path, "pole" == data_type)
    if 1:
        print("make_instance_segmentation_data")
        pre_process.make_instance_segmentation_data()
    if args.list:
        print("make_train_val_file_list")
        pre_process.make_train_val_file_list()
    if args.anchor:
        print("output_bbox_width_and_height")
        pre_process.output_bbox_width_and_height()
