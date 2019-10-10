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

# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import mxnet as mx
import random
import cv2
import numpy as np
import time
import traceback


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


def write_list(path_out, image_list, rec_image_list):
    with open(path_out, 'a') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '{}\t'.format(j)
            label = rec_image_list[item[1]]
            line += '{}\t{}\n'.format(item[1], label)
            fout.write(line)


def make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks, taskid):
    image_label_map = {}
    image_list = []

    dir_path = sour_data
    if taskid:
        index = []
        data_paths = os.path.dirname(data_path)
        lst_list = os.listdir(data_paths)
        for i in lst_list:
            if i.endswith(".lst"):
                with open(os.path.join(data_paths, i), 'r') as f:
                    while True:
                        line = f.readline()
                        if not line:
                            break
                        line = [i.strip() for i in line.strip().split('\t')]
                        index.append(int(line[0]))

        index.sort()
        image_index = index[-1]
    else:
        image_index = 0

    package_dir_list = os.listdir(dir_path)
    for package_id in package_dir_list:
        # if not package_id.isdigit():
        #     continue

        file_list = os.listdir(os.path.join(dir_path, package_id))
        for file_id in file_list:
            if file_id.startswith("label") or file_id.endswith("png"):
                continue

            image_path = os.path.join(dir_path, package_id, file_id)
            if file_id.endswith("jpg"):
                label_path = image_path[:-3] + "png"
                if os.path.exists(label_path):
                    image_list.append((image_index, image_path, "0"))
                    image_index += 1
                    image_label_map[image_path] = label_path
                else:
                    print("No have label_path: {}".format(image_path))

    # image_list = list_image(sour_data, recursive, exts)
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


def read_list(path_in):
    with open(path_in) as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            line = [i.strip() for i in line.strip().split('\t')]
            line_len = len(line)
            if line_len < 4:
                print('lst should at least has three parts, but only has %s parts for %s' % (line_len, line))
                continue
            try:
                item = [int(line[0])] + [float(line[1])] + [i for i in line[2:]]
            except Exception as e:
                print('Parsing lst met error for %s, detail: %s' % (line, e))
                continue
            yield item


def image_encode(pass_through, center_crop, center_pad, resize, quality, i, item, q_out):
    fullpath = item[2]

    if not os.path.exists(fullpath):
        print(fullpath)

    # if len(item) > 3 and pack_label:
    #     header = mx.seg_recordio.ISegRHeader(0, 0, 0, 0, item[0], 0)
    # else:
    #     header = mx.seg_recordio.ISegRHeader(0, 0, 0, 0, item[0], 0)

    if pass_through:
        try:
            img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
            ret, buf = cv2.imencode(".jpg", img)
            assert ret, 'failed to encode image'
            image_data = buf.tostring()
            image_len = len(image_data)

            label_path = item[-1]
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            ret, buf = cv2.imencode(".png", label)
            assert ret, 'failed to encode label'
            label_data = buf.tostring()
            label_len = len(label_data)

            header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, item[0], 0)

            s = mx.seg_recordio.pack(header, image_data, label_data)
            q_out.put((i, s, item))
        except Exception as e:
            traceback.print_exc()
            print('pack_img error:', item[1], e)
            q_out.put((i, None, item))
        return

    try:
        img = cv2.imread(fullpath, cv2.IMREAD_COLOR)
        label_path = item[-1]
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        ret, buf = cv2.imencode(".jpg", img)
        assert ret, 'failed to encode image'
        image_data = buf.tostring()
        image_len = len(image_data)

        ret, buf = cv2.imencode(".png", label)
        assert ret, 'failed to encode label'
        label_data = buf.tostring()
        label_len = len(label_data)
        # with open(fullpath, "r") as f:
        #     s = f.read()
        #     image_len = len(s)
        # with open(label_path, "r") as f1:
        #     s1 = f1.read()
        #     label_len = len(s1)

        header = mx.seg_recordio.ISegRHeader(0, 0, image_len, label_len, item[0], 0)
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
            label = label[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2
            img = img[:, margin:margin + img.shape[0]]
            label = label[:, margin:margin + label.shape[0]]
    if center_pad:
        newsize = max(img.shape[:2])
        new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
        new_label = np.ones((newsize, newsize) + label.shape[2:], np.uint8) * 127
        margin0 = (newsize - img.shape[0]) // 2
        margin1 = (newsize - img.shape[1]) // 2
        new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
        new_label[margin0:margin0 + label.shape[0], margin1:margin1 + label.shape[1]] = label
        img = new_img
        label = new_label
    if resize:
        if img.shape[0] > img.shape[1]:
            newsize = (resize, img.shape[0] * resize // img.shape[1])
        else:
            newsize = (img.shape[1] * resize // img.shape[0], resize)
        img = cv2.resize(img, newsize)
        label = cv2.resize(label, newsize)

    try:
        s = mx.seg_recordio.pack_img(header, img, label, quality=quality, img_fmt='.jpg', label_fmt='.png')
        q_out.put((i, s, item))
    except Exception as e:
        traceback.print_exc()
        print('pack_img error on file: %s' % fullpath, e)
        q_out.put((i, None, item))
        return


def read_worker(pass_through, center_crop, center_pad, resize, quality, q_in, q_out):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(pass_through, center_crop, center_pad, resize, quality, i, item, q_out)


def write_worker(q_out, fname, working_dir, lock, cur_count):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.seg_recordio.MXIndexedSegRecordIO(
        os.path.join(working_dir, fname_idx), os.path.join(working_dir, fname_rec), 'w')
    buf = {}
    more = True
    while more:
        deq = q_out.get()
        if deq is not None:
            i, s, item = deq
            buf[i] = (s, item)
        else:
            more = False
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
            lock.acquire()
            cur_count.value += 1
            lock.release()
