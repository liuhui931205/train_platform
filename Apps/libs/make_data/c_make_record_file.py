# -*- coding:utf-8 -*-
from __future__ import print_function
import os
# import sys
# sys.path.insert(0, "/opt/densenet.mxnet")

import numpy as np
import mxnet as mx
import random
import cv2
import time
import traceback
from Apps.utils.utils import arrow_labels as arrow_labels_v2

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


def write_list(path_out, image_list):
    with open(path_out, 'w') as fout:
        for i, item in enumerate(image_list):
            line = '%d\t' % item[0]
            for j in item[2:]:
                line += '{}\t'.format(j)
            line += '%s\n' % item[1]
            fout.write(line)


def make_list(data_path, sour_data, chunks, train_ratio, test_ratio, shuffle):
    class_id_map = {label.id: label.categoryId for label in arrow_labels_v2}

    image_list = []
    image_index = 0

    dir_path = sour_data
    label_file = os.path.join(dir_path, "ImageType.csv")
    if not os.path.exists(label_file):
        return

    with open(label_file, "r") as f:
        line_str = f.readline()
        # skip first line
        line_str = f.readline()
        while line_str:
            line_str = line_str.strip()
            file_name, class_id = line_str.split(",")
            real_path = os.path.join(dir_path, file_name)

            class_id = int(class_id)
            map_id = class_id_map[class_id]

            if class_id > 1 or map_id > 1:
                print("wait")

            image_list.append((image_index, real_path, str(map_id)))

            image_index += 1
            line_str = f.readline()

    # image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if shuffle is True:
        random.seed(100)
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
            write_list(data_path + str_chunk + '.lst', chunk)
        else:
            if test_ratio:
                write_list(data_path + str_chunk + '_test.lst', chunk[:sep_test])
            if train_ratio + test_ratio < 1.0:
                write_list(data_path + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(data_path + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])


def make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks):
    class_id_map = {label.id: label.categoryId for label in arrow_labels_v2}

    image_list = []
    image_index = 0

    dir_path = sour_data

    class_dir_list = os.listdir(dir_path)
    for class_id in class_dir_list:
        file_list = os.listdir(os.path.join(dir_path, class_id))
        for file_id in file_list:
            if len(file_id) < 4 or file_id[-3:] not in ['jpg', 'png']:
                continue

            real_path = os.path.join(dir_path, class_id, file_id)
            class_num = int(class_id)
            map_id = class_id_map[class_num]

            image_list.append((image_index, real_path, str(map_id)))
            image_index += 1

    # image_list = list_image(args.root, args.recursive, args.exts)
    image_list = list(image_list)
    if shuffle is True:
        random.seed(100)
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
            write_list(data_path + str_chunk + '.lst', chunk)
        else:
            if test_ratio:
                write_list(data_path + str_chunk + '_test.lst', chunk[:sep_test])
            if train_ratio + test_ratio < 1.0:
                write_list(data_path + str_chunk + '_val.lst', chunk[sep_test + sep:])
            write_list(data_path + str_chunk + '_train.lst', chunk[sep_test:sep_test + sep])


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


def image_encode(sour_data, pass_through, center_crop, center_pad, resize, quality, i, item, q_out, pack_label, color,
                 encoding):
    fullpath = os.path.join(sour_data, item[1])

    if not os.path.exists(fullpath):
        print(fullpath)

    if len(item) > 3 and pack_label:
        header = mx.recordio.IRHeader(0, item[2:], item[0], 0)
    else:
        header = mx.recordio.IRHeader(0, item[2], item[0], 0)

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
        # img = cv2.imread(fullpath, args.color)
        img = cv2.imread(fullpath, color)
        img = img[:, :, ::-1]
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
            margin = (img.shape[0] - img.shape[1]) // 2;
            img = img[margin:margin + img.shape[1], :]
        else:
            margin = (img.shape[1] - img.shape[0]) // 2;
            img = img[:, margin:margin + img.shape[0]]
    if center_pad:
        newsize = max(img.shape[:2])
        new_img = np.ones((newsize, newsize) + img.shape[2:], np.uint8) * 127
        margin0 = (newsize - img.shape[0]) // 2
        margin1 = (newsize - img.shape[1]) // 2
        new_img[margin0:margin0 + img.shape[0], margin1:margin1 + img.shape[1]] = img
        img = new_img
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


def read_worker(q_in, sour_data, pass_through, center_crop, center_pad, resize, quality, q_out, pack_label, color,
                encoding):
    while True:
        deq = q_in.get()
        if deq is None:
            break
        i, item = deq
        image_encode(sour_data, pass_through, center_crop, center_pad, resize, quality, i, item, q_out, pack_label,
                     color, encoding)


def write_worker(q_out, fname, working_dir, lock, cur_count):
    pre_time = time.time()
    count = 0
    fname = os.path.basename(fname)
    fname_rec = os.path.splitext(fname)[0] + '.rec'
    fname_idx = os.path.splitext(fname)[0] + '.idx'
    record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                           os.path.join(working_dir, fname_rec), 'w')
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

            if count % 1000 == 0:
                cur_time = time.time()
                print('time:', cur_time - pre_time, ' count:', count)
                pre_time = cur_time
            count += 1
            lock.acquire()
            cur_count.value += 1
            lock.release()
