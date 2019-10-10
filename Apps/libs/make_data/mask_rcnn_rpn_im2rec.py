#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import os
import sys

curr_path = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(curr_path, "../python"))
import mxnet as mx
import random
import argparse
import cv2
import time
import traceback

import PIL.Image as Image
import numpy as np
from config import class_id

try:
    import multiprocessing
except ImportError:
    multiprocessing = None


def list_image(sour_data, recursive, exts):
    i = 0
    if recursive:
        cat = {}
        for path, dirs, files in os.walk(sour_data, followlinks=True):
            dirs.sort()
            files.sort()
            for fname in files:
                fpath = os.path.join(path, fname)
                suffix = os.path.splitext(fname)[1].lower()
                if os.path.isfile(fpath) and (suffix in exts):
                    if path not in cat:
                        cat[path] = len(cat)
                    yield (i, os.path.relpath(fpath, sour_data), cat[path])
                    i += 1
        for k, v in sorted(cat.items(), key=lambda x: x[1]):
            print(os.path.relpath(k, sour_data), v)
    else:
        for fname in sorted(os.listdir(sour_data)):
            fpath = os.path.join(sour_data, fname)
            suffix = os.path.splitext(fname)[1].lower()
            if os.path.isfile(fpath) and (suffix in exts):
                yield (i, os.path.relpath(fpath, sour_data), 0)
                i += 1


# def write_list(path_out, image_list):
#     with open(path_out, 'w') as fout:
#         for i, item in enumerate(image_list):
#             line = '%d\t' % item[0]
#             for j in item[2:]:
#                 line += '%f\t' % j
#             line += '%s\n' % item[1]
#             fout.write(line)


#  kuandeng for maskrcnn
def write_list(path_out, image_list, rec_image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '{}\t'.format(j)
            label = rec_image_list[item[1]]
            line += '{}\t{}\n'.format(item[1], label)
            fout.write(line)


# kuandeng for maskrcnn
def make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks, taskid=None):
    image_label_map = {}
    image_list = []

    dir_path = sour_data
    image_index = 0

    package_dir_list = os.listdir(dir_path)
    for package_id in package_dir_list:
        if not package_id.isdigit():
            # continue
            pass

        file_list = os.listdir(os.path.join(dir_path, package_id))
        for file_id in file_list:
            if file_id.startswith("label") or file_id.endswith("png"):
                continue

            image_path = os.path.join(dir_path, package_id, file_id)
            if file_id.endswith("jpg"):
                # label_path = image_path[:-3] + "png"
                label_path = image_path[:-4] + "-ins.png"
                if os.path.exists(label_path):
                    image_list.append((image_index, image_path, "0"))
                    image_index += 1
                    image_label_map[image_path] = label_path
                else:
                    print("No have label_path: {}".format(image_path))

    # image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if shuffle is True:
        random.seed(random.random())
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + chunks - 1) // chunks
    for i in range(chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * train_ratio)
        sep_test = int(chunk_size * test_ratio)
        if train_ratio == 1.0:
            write_list(data_path + str_chunk + '.lst', chunk, image_label_map)
        else:
            if test_ratio:
                write_list(data_path + str_chunk + '_test.lst', chunk[:sep_test], image_label_map)
            if train_ratio + test_ratio < 1.0:
                write_list(data_path + str_chunk + '_val.lst', chunk[sep_test + sep:], image_label_map)
            write_list(data_path + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep], image_label_map)
    print("Image Num: ", N)


def make_list(args):
    image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if args.shuffle is True:
        random.seed(100)
        random.shuffle(image_list)
    N = len(image_list)
    chunk_size = (N + args.chunks - 1) // args.chunks
    for i in range(args.chunks):
        chunk = image_list[i * chunk_size:(i + 1) * chunk_size]
        if args.chunks > 1:
            str_chunk = '_%d' % i
        else:
            str_chunk = ''
        sep = int(chunk_size * args.train_ratio)
        sep_test = int(chunk_size * args.test_ratio)
        if args.train_ratio == 1.0:
            write_list(args.prefix + str_chunk + '.lst', chunk)
        else:
            if args.test_ratio:
                write_list(args.prefix + str_chunk + '_test.lst', chunk[:sep_test])
            if args.train_ratio + args.test_ratio < 1.0:
                write_list(args.prefix + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(args.prefix + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 2:
                print('lst should at least has two parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                # item = [int(line[0])] + [line[-1]] + [float(i) for i in line[1:-1]]
                # index 0 image_path label_path
                item = [int(line[0])] + [float(line[1])] + [i for i in line[2:]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


# class_id = [0, 1, 2]
# bbox: [cat_id, xmin, ymin, xmax, ymax, 0, cat_id, xmin, ymin, xmax, ymax, 0]
def load_from_det_image(ins_det_path, class_id=class_id, min_bbox_area=400):
    assert os.path.exists(ins_det_path), 'Path does not exist: {}'.format(ins_det_path)
    im = Image.open(ins_det_path)
    pixel = list(im.getdata())
    pixel = np.array(pixel).reshape([im.size[1], im.size[0]])
    boxes = []
    for c in range(1, len(class_id)):
        px = np.where((pixel >= class_id[c] * 1000) & (pixel < (class_id[c] + 1) * 1000))
        if len(px[0]) == 0:
            continue
        ids = np.unique(pixel[px])
        for id in ids:
            px = np.where(pixel == id)
            x_min = np.min(px[1])
            y_min = np.min(px[0])
            x_max = np.max(px[1])
            y_max = np.max(px[0])
            if x_max - x_min <= 1 or y_max - y_min <= 1:
                continue
            if (x_max - x_min) * (y_max - y_min) < min_bbox_area:
                continue
            boxes = boxes + [c, x_min, y_min, x_max, y_max, 0]
            # boxes.append([c, x_min, y_min, x_max, y_max, 0])
    return 6, boxes


#  index 0 image_path label_path
def image_encode(count, lock,  pass_through, center_crop, encoding, resize, quality, color, i, item, q_out):
    fullpath = item[2]
    label_path = item[3]
    assert os.path.exists(fullpath), 'Image path does not exist: {}'.format(fullpath)
    assert os.path.exists(label_path), 'Label path does not exist: {}'.format(label_path)

    # print("Processing({}): {}".format(i, fullpath))
    # print("Processing({})".format(i), end = " ")

    # bbox: [cat_id, xmin, ymin, xmax, ymax, 0, cat_id, xmin, ymin, xmax, ymax, 0]
    label_dim, bbox = load_from_det_image(label_path, class_id=class_id)
    if len(bbox) == 0:
        print("no object: ", label_path)
        with lock:
            count.value += 1
        return

    header = mx.recordio.IRHeader(0, [2, label_dim] + ["{0:.4f}".format(x) for x in bbox], item[0], 0)

    if pass_through:
        try:
            with open(fullpath, 'rb') as fin:
                img = fin.read()
            s = mx.recordio.pack(header, img)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, color)
    except:
        traceback.print_exc()
        print('imread error trying to load file: %s ' % fullpath)
        q_out.put((i, None, item))
        return
    if img is None:
        print('imread read blank (None) image for file: %s' % fullpath)
        q_out.put((i, None, item))
        return
    if center_crop:
        if img.shape[0] > img.shape[1]:
            margin = (img.shape[0] - img.shape[1]) // 2
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]
    if resize:
        if img.shape[0] > img.shape[1]:
            newsize = (resize, img.shape[0] * resize // img.shape[1])
        else:
            newsize = (img.shape[1] * resize // img.shape[0], resize)
        img = cv2.resize(img, newsize)

    try:
        s = mx.recordio.pack_img(header, img, quality=quality, img_fmt=encoding)
        q_out.put((i, s, item))

    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(count, lock, pass_through, center_crop, encoding, resize, quality, color, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(count, lock, pass_through, center_crop, encoding, resize, quality, color, i, item, q_out)


def write_worker(q_out, fname, working_dir,lock, cur_count):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(
        os.path.join(working_dir, fname_idx), os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            if s is not None:
                record.write_idx(item[0], s)

            if count % 50 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1
            lock.acquire()
            cur_count.value += 1
            lock.release()

            # buf[i] = (s, item)
            # print("%%%%: {}".format(i))

        else:
            # print("finish: ", count)
            more = False
        """
        while count in buf:
            s, item = buf[count]
            del buf[count]
            if s is not None:
                record.write_idx(item[0], s)

            if count % 100 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1
        """
    print('***** time:', time.time() - pre_time, ' count:', count)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description='Create an image list or \
        make a record database by reading from an image list')
    parser.add_argument('prefix', help='prefix of input/output lst and rec files.')
    parser.add_argument('root', help='path to folder containing images.')

    cgroup = parser.add_argument_group('Options for creating image lists')
    cgroup.add_argument(
        '--list',
        action='store_true',
        help='If this is set im2rec will create image list(s) by traversing root folder\
        and output to <prefix>.lst.\
        Otherwise im2rec will read <prefix>.lst and create a database at <prefix>.rec')
    cgroup.add_argument(
        '--exts', nargs='+', default=['.jpeg', '.jpg', '.png'], help='list of acceptable image extensions.')
    cgroup.add_argument('--chunks', type=int, default=1, help='number of chunks.')
    cgroup.add_argument('--train-ratio', type=float, default=1.0, help='Ratio of images to use for training.')
    cgroup.add_argument('--test-ratio', type=float, default=0, help='Ratio of images to use for testing.')
    cgroup.add_argument(
        '--recursive',
        action='store_true',
        help='If true recursively walk through subdirs and assign an unique label\
        to images in each folder. Otherwise only include images in the root folder\
        and give them label 0.')
    cgroup.add_argument(
        '--no-shuffle',
        dest='shuffle',
        action='store_false',
        help='If this is passed, \
        im2rec will not randomize the image order in <prefix>.lst')
    rgroup = parser.add_argument_group('Options for creating database')
    rgroup.add_argument(
        '--pass-through', action='store_true', help='whether to skip transformation and save image as is')
    rgroup.add_argument(
        '--resize',
        type=int,
        default=0,
        help='resize the shorter edge of image to the newsize, original images will\
        be packed by default.')
    rgroup.add_argument(
        '--center-crop', action='store_true', help='specify whether to crop the center image to make it rectangular.')
    rgroup.add_argument(
        '--quality',
        type=int,
        default=95,
        help='JPEG quality for encoding, 1-100; or PNG compression for encoding, 1-9')
    rgroup.add_argument(
        '--num-thread',
        type=int,
        default=1,
        help='number of thread to use for encoding. order of images will be different\
        from the input list if >1. the input list will be modified to match the\
        resulting order.')
    rgroup.add_argument(
        '--color',
        type=int,
        default=1,
        choices=[-1, 0, 1],
        help='specify the color mode of the loaded image.\
        1: Loads a color image. Any transparency of image will be neglected. It is the default flag.\
        0: Loads image in grayscale mode.\
        -1:Loads image as such including alpha channel.')
    rgroup.add_argument(
        '--encoding', type=str, default='.jpg', choices=['.jpg', '.png'], help='specify the encoding of the images.')
    rgroup.add_argument(
        '--pack-label', action='store_true', help='Whether to also pack multi dimensional label in the record file')
    args = parser.parse_args()
    args.prefix = os.path.abspath(args.prefix)
    args.root = os.path.abspath(args.root)
    return args
