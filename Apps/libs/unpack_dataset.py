# -*- coding:utf-8 -*-
import os
import shutil
import sys
from config import data0_sour_data

reload(sys)
sys.setdefaultencoding("utf-8")


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 3:
                print('lst should at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def start_unpack(lst_file, check_dir):
    image_list = read_list(lst_file)
    image_list = list(image_list)
    count = len(image_list)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    if str(lst_file).endswith("train.lst"):
        check_dir = os.path.join(check_dir, "train")
    elif str(lst_file).endswith("val.lst"):
        check_dir = os.path.join(check_dir, "val")
    elif str(lst_file).endswith("test.lst"):
        check_dir = os.path.join(check_dir, "test")
    if os.path.exists(check_dir):
        shutil.rmtree(check_dir)
    class_dict = {}
    for label_info in image_list:
        label_path = label_info[1]
        label_id = int(label_info[2])

        if label_id not in class_dict:
            class_dict[label_id] = []
        class_dict[label_id].append(label_path)

        class_dir = os.path.join(check_dir, str(label_id))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        image_path = os.path.join(class_dir, os.path.basename(label_path))
        try:
            shutil.copyfile(label_path, image_path)
        except Exception as e:
            print(e)

    for label_id, labels in class_dict.items():
        print ("{}: {}/{}".format(str(label_id), str(len(labels)), count))


def start_unpack_d(lst_file, check_dir):
    image_list = read_list(lst_file)
    image_list = list(image_list)
    count = len(image_list)
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

    if str(lst_file).endswith("train.lst"):
        check_dir = os.path.join(check_dir, "train")
    elif str(lst_file).endswith("val.lst"):
        check_dir = os.path.join(check_dir, "val")
    elif str(lst_file).endswith("test.lst"):
        check_dir = os.path.join(check_dir, "test")
    if os.path.exists(check_dir):
        shutil.rmtree(check_dir)
    class_dict = {}
    for label_info in image_list:
        label_path = label_info[1]
        label_id = int(label_info[2])

        if label_id not in class_dict:
            class_dict[label_id] = []
        class_dict[label_id].append(label_path)

        class_dir = os.path.join(check_dir, str(label_id))
        if not os.path.exists(class_dir):
            os.makedirs(class_dir)

        image_path = os.path.join(class_dir, os.path.basename(label_path))

        try:
            q = label_path.split('/')
            d = os.path.join(data0_sour_data, q[-3], q[-2], q[-1])
            shutil.copyfile(d, image_path)

        except Exception as e:
            print(e)
        # shutil.copy(label_path, image_path)

    for label_id, labels in class_dict.items():
        print ("{}: {}/{}".format(str(label_id), str(len(labels)), count))
