# -*- coding:utf-8 -*-
import os
import json
import time
import multiprocessing
import shutil
import cv2
import numpy as np
from flask import current_app
from Apps.utils.copy_all import copyFiles
from Apps.utils.utils import Task
from Apps.utils.utils import self_full_labels
from config import GlobalVar
from Apps.modules.base_process_label import BaseProcessLabel
from Apps.utils.client import client


class ProcessLabelUnionHandler(BaseProcessLabel):

    def __init__(self):
        super(BaseProcessLabel, self).__init__()
        self.file_list = list()
        self.pixel = 50
        self.no_qualified_dict = None
        self.task_id = ''
        self.status = ''
        self.dir_path = []
        self.src_dir = GlobalVar.check_dir.value
        self.temp_dir = GlobalVar.check_dir.value + "_full"
        self.dest_dir = GlobalVar.release_dir.value
        self.total_count = None
        if not os.path.exists(self.src_dir):
            os.makedirs(self.src_dir)
        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

        manager = multiprocessing.Manager()
        self.lock = multiprocessing.Lock()
        self.remote_process_queue = manager.Queue()
        self.remote_process_count = manager.Value("i", 0)
        self.full_cut_queue = manager.Queue()
        self.remote_cut_count = manager.Value("i", 0)

    def start_process(self, version, types, task_id, status, color_info):
        self.task_id = task_id
        pro = multiprocessing.Process(target=self.get, args=(version,))
        name = time.strftime("full-aug-%Y%m%d", time.localtime())
        self.starts(version, name, types, task_id, status, color_info)
        pro.start()
        self.error_code = 1
        self.message = 'success'

    def get(self, version="all"):
        time1 = time.time()
        try:
            _ver = version
            self.src_dir = os.path.join(self.src_dir, _ver)
            self.temp_dir = os.path.join(self.temp_dir, _ver)

            if not os.path.exists(self.src_dir):
                task_count = 0
            else:
                if not os.path.exists(self.temp_dir):
                    os.makedirs(self.temp_dir)
                dir_list = os.listdir(self.src_dir)
                for dir in dir_list:
                    if not dir.isdigit() or not os.path.isdir(os.path.join(self.src_dir, str(dir))):
                        continue
                    csv_name = "ext-" + str(dir) + ".csv"
                    csv_file = os.path.join(self.src_dir, str(dir), csv_name)
                    if not os.path.exists(csv_file):
                        self.dir_path.append(os.path.join(self.src_dir, str(dir)))
                        csv_file = None

                    self.clean_dir(_csv_file=csv_file)
                    self.prepare_crop(_csv_file=csv_file)

                task_count = self.full_cut_queue.qsize()
                task_count = task_count // 2
                self.total_count = task_count
                task = Task(None, None, None, None, True)
                self.full_cut_queue.put(task)
                self.do_work()
                dir_list = os.listdir(self.src_dir)
                for i in dir_list:
                    if not i.isdigit() or not os.path.isdir(os.path.join(self.src_dir, str(i))):
                        continue

                    _sub_dir = os.path.join(self.src_dir, str(i))
                    if not os.path.exists(_sub_dir):
                        continue

                    dest_dir = os.path.join(self.temp_dir, str(i))
                    if not os.path.exists(dest_dir):
                        os.makedirs(dest_dir)

                    origin_list = os.listdir(_sub_dir)

                    for _image in origin_list:
                        _image = _image.strip()
                        if not _image.startswith("label-"):
                            continue
                        if not _image.endswith("png"):
                            continue
                        # start with label-
                        anna_file = _image[6:]
                        origin_name = anna_file[:-3] + "jpg"

                        if origin_name in GlobalVar.filter_data:
                            current_app.logger.info("filter image:{}".format(origin_name))
                            continue

                        image_path = os.path.join(_sub_dir, _image)
                        origin_image = os.path.join(_sub_dir, origin_name)
                        if not os.path.exists(origin_image):
                            os.remove(image_path)
                            current_app.logger.info("package:{}, file missing:[{}]=>[{}]".format(
                                str(i), origin_name, _image))
                            continue

                        result_path = os.path.join(self.temp_dir, str(i), anna_file)
                        task = Task(str(i), image_path, result_path, origin_image, False)
                        self.remote_process_queue.put(task)
                        self.remote_process_count.value += 1
                for i in range(20):
                    task = Task(None, None, None, None, True)
                    self.remote_process_queue.put(task)

                all_processes = []
                self.no_qualified_dict = multiprocessing.Manager().dict()
                for i in range(20):
                    process = multiprocessing.Process(target=self.transform, args=(self.no_qualified_dict,))
                    process.daemon = True
                    all_processes.append(process)
                count = multiprocessing.Process(target=self.update_task)
                count.start()
                for process in all_processes:
                    process.start()

                for process in all_processes:
                    process.join()

                # 拷贝
                if _ver != "all":
                    cur_day = time.strftime("full-aug-%Y%m%d", time.localtime())
                else:
                    cur_day = time.strftime("full-all-%Y%m%d", time.localtime())
                copy_dir = self.temp_dir
                dir_list = os.listdir(copy_dir)
                dest_scp, dest_sftp, dest_ssh = client(id="host")
                files = dest_sftp.listdir(path=self.dest_dir)
                self.dest_dir = os.path.join(self.dest_dir, cur_day)
                if cur_day not in files:
                    dest_sftp.mkdir(self.dest_dir)

                for _dir in dir_list:
                    old_src = os.path.join(copy_dir, _dir)
                    dest_scp.put(old_src, self.dest_dir, recursive=True)
                # copy_dir = self.temp_dir
                # files = os.listdir(self.dest_dir)
                # self.dest_dir = os.path.join(self.dest_dir, cur_day)
                # if cur_day not in files:
                #     os.mkdir(self.dest_dir)
                # copyFiles(copy_dir, self.dest_dir)
                count.join()
                time2 = time.time()
                result_obj = {
                    "count": str(task_count),
                    "qualified_count": str(task_count - len(self.no_qualified_dict)),
                    "no_qualified_count": str(len(self.no_qualified_dict)),
                    "no_qualified_detail": dict(self.no_qualified_dict),
                    "time": str(time2 - time1) + " s"
                }
                json.dump(result_obj, open(os.path.join(self.src_dir, 'info.json'), 'w'))
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()

        except Exception as err:
            print(err)

    def prepare_crop(self, _csv_file):
        if _csv_file:
            root_dir = os.path.dirname(_csv_file)
            csv_name = os.path.basename(_csv_file)

            name_list = csv_name.split(".")
            csv_name = name_list[0]
            if csv_name.startswith("ext-"):
                csv_name = csv_name[4:]
            package_index = csv_name

            with open(_csv_file, "r") as f:
                line_str = f.readline()

                while line_str:
                    image, label = line_str.split(",")
                    image = image.strip()
                    label = label.strip()

                    image_name = str(image)
                    if image_name.startswith("ext"):
                        image_name = image_name[4:]
                    # print(global_variables.filter_data)
                    if image_name in GlobalVar.filter_data:
                        line_str = f.readline()
                        continue

                    if not image.startswith("ext-") or not label.startswith("ext-"):
                        line_str = f.readline()
                        continue

                    if not os.path.exists(os.path.join(root_dir, label)):
                        line_str = f.readline()
                        current_app.logger.info("package:{}, label missing:[{}]=>[{}]".format(
                            package_index, image, label))
                        continue

                    if not os.path.exists(os.path.join(root_dir, image)):
                        line_str = f.readline()
                        current_app.logger.info("package:{}, image missing:[{}]=>[{}]".format(
                            package_index, image, label))
                        continue

                    origin_image = os.path.join(root_dir, image[4:])
                    _src = os.path.join(root_dir, image)
                    _dest_image = os.path.join(root_dir, origin_image)
                    task = Task(package_index, _src, _dest_image, None)
                    self.full_cut_queue.put(task)
                    self.remote_cut_count.value += 1

                    _src = os.path.join(root_dir, label)
                    _dest_id2 = label[4:]
                    _dest_label = os.path.join(root_dir, _dest_id2)
                    _dest_label = _dest_label.strip()

                    task = Task(package_index, _src, _dest_label, None)
                    self.full_cut_queue.put(task)
                    self.remote_cut_count.value += 1
                    line_str = f.readline()
        else:
            for dir in self.dir_path:
                package_index = os.path.basename(dir)
                lis = os.listdir(dir)
                for i in lis:
                    if i.startswith('ext-') and i.endswith('png'):
                        continue
                    if i.startswith('label-') and i.endswith('png'):
                        continue
                    if i.startswith('ext-') and i.endswith('jpg'):
                        name = i[4:-11]
                        image = i
                        label = 'ext-label-' + name + '_00_004.png'

                    if i[0].isdigit() and i.endswith('jpg'):
                        name = i[:-11]
                        image = i
                        label = 'label-' + name + '_00_004.png'
                    image_name = str(image)
                    if image_name.startswith("ext"):
                        image_name = image_name[4:]
                    # print(global_variables.filter_data)
                    if image_name in GlobalVar.filter_data:
                        continue

                    # if not image.startswith("ext-") or not label.startswith("ext-"):
                    #     continue

                    if not os.path.exists(os.path.join(dir, label)):
                        current_app.logger.info("package:{}, label missing:[{}]=>[{}]".format(
                            package_index, image, label))
                        continue

                    if not os.path.exists(os.path.join(dir, image)):
                        current_app.logger.info("package:{}, image missing:[{}]=>[{}]".format(
                            package_index, image, label))
                        continue

                    origin_image = os.path.join(dir, image)
                    _src = os.path.join(dir, image)
                    _dest_image = os.path.join(dir, origin_image)
                    task = Task(package_index, _src, _dest_image, None)
                    self.full_cut_queue.put(task)
                    self.remote_cut_count.value += 1

                    _src = os.path.join(dir, label)
                    _dest_id2 = label
                    _dest_label = os.path.join(dir, _dest_id2)
                    _dest_label = _dest_label.strip()

                    task = Task(package_index, _src, _dest_label, None)
                    self.full_cut_queue.put(task)
                    self.remote_cut_count.value += 1

    def do_work(self):
        if self.full_cut_queue.empty():
            return

        while not self.full_cut_queue.empty():
            task = self.full_cut_queue.get()

            if not isinstance(task, Task):
                break

            if task.exit_flag:
                break

            self.remote_cut_count.value -= 1

            time1 = time.time()

            _src = task.src_path
            _dest = task.dest_path

            img = cv2.imread(_src)
            if img is None:
                current_app.logger.info("image is none[{}/{}]".format(task.package_index, _src))
                continue
            width = img.shape[1]
            height = img.shape[0]

            if width == 2448 and height == 1024:
                continue
            if width == 2448 and height == 2048:
                crop_img = img[0:height, 0:width]
            else:
                crop_img = img[self.pixel:height - self.pixel, self.pixel:width - self.pixel]
            cv2.imwrite(_dest, crop_img)

            time2 = time.time()
            current_app.logger.info("process[{}/{}] in {} s".format(task.package_index, _src, time2 - time1))

    def transform(self, dicts):
        while not self.remote_process_queue.empty():
            task = self.remote_process_queue.get()

            if not isinstance(task, Task):
                break

            if task.exit_flag:
                break
            self.lock.acquire()
            self.remote_process_count.value -= 1
            self.lock.release()
            image_path = task.src_path
            result_path = task.dest_path
            origin_image_path = task.dest_label

            time1 = time.time()

            img = cv2.imread(image_path)
            if img is None:
                current_app.logger.info("image is none[{}/{}]".format(task.package_index, image_path))
                continue

            width = img.shape[1]
            height = img.shape[0]

            other_category = 255
            label_data = np.zeros((height, width), np.uint8)
            for label in self_full_labels:
                if label.name == u"Ignore":
                    other_category = label.categoryId
                    break
            label_data[0:height, 0:width] = other_category

            for label in self_full_labels:
                color = (label.color[2], label.color[1], label.color[0])
                label_data[np.where((img == color).all(axis=2))] = label.categoryId

            # 校验"Ignore"类别的占比
            check_data = label_data[height // 2:height, 0:width]
            other_count = np.sum(check_data == other_category)
            valid_count = width * height * 0.01
            if other_count > valid_count:
                label_name = os.path.basename(image_path)
                file_name = label_name.split(".")[:-1]
                file_name = ".".join(file_name)
                if file_name.startswith("ext-"):
                    file_name = file_name[4:]
                if file_name.startswith("label-"):
                    file_name = file_name[6:]
                origin_image_path = os.path.join(os.path.dirname(image_path), file_name + ".jpg")
                current_app.logger.info("label[{}/{}] not qualified".format(task.package_index, image_path))
                dicts[label_name] = task.package_index

                if os.path.exists(origin_image_path):
                    os.remove(origin_image_path)
                if os.path.exists(image_path):
                    os.remove(image_path)
            else:
                cv2.imwrite(result_path, label_data)
                dest_image_path = result_path[:-3] + "jpg"
                shutil.copy(origin_image_path, dest_image_path)
                dest_label_path = os.path.join(os.path.dirname(result_path), os.path.basename(image_path))
                shutil.copy(image_path, dest_label_path)

            time2 = time.time()
            current_app.logger.info("process[{}/{}] in {} s".format(task.package_index, image_path, time2 - time1))

    def clean_dir(self, _csv_file):
        image_list = []
        image_path = ''
        label_path = ''
        image = ''
        label = ''
        if _csv_file:
            package_index = ""
            root_dir = os.path.dirname(_csv_file)
            csv_name = os.path.basename(_csv_file)
            image_list.append(str(csv_name).strip())

            if csv_name.startswith("ext-"):
                origin_csv = csv_name[4:]
                image_list.append(str(origin_csv).strip())

                name_list = origin_csv.split(".")
                new_file_name = name_list[0] + ".txt"
                package_index = name_list[0]
                image_list.append(new_file_name)

            with open(_csv_file, "r") as f:
                line_str = f.readline()

                while line_str:
                    image, label = line_str.split(",")
                    image = str(image).strip()
                    label = str(label).strip()

                    image_path = os.path.join(root_dir, image)
                    label_path = os.path.join(root_dir, label)

                    if os.path.exists(image_path) and os.path.exists(label_path):
                        image_list.append(image)
                        image_list.append(label)

                        if image.startswith("ext-"):
                            origin_image = image[4:]
                            image_list.append(str(origin_image).strip())

                            annotation_file = origin_image[:-3] + "png"
                            image_list.append(str(annotation_file).strip())

                        if label.startswith("ext-"):
                            origin_label = label[4:]
                            image_list.append(str(origin_label).strip())

                    if not os.path.exists(image_path):
                        if os.path.exists(label_path):
                            current_app.logger.info("package:{}, not match:[{}] not exist=>[{}] exist".format(
                                package_index, image, label))
                            if os.path.exists(label_path):
                                os.remove(label_path)
                        else:
                            current_app.logger.info("package:{}, file missing:[{}]=>[{}]".format(
                                package_index, image, label))
                    else:
                        if not os.path.exists(label_path):
                            current_app.logger.info("package:{}, not match:[{}] exist=>[{}] not exist".format(
                                package_index, image, label))
                            if os.path.exists(image_path):
                                os.remove(image_path)

                    line_str = f.readline()

            file_list = os.listdir(root_dir)
            for file_id in file_list:
                file_id = str(file_id).strip()

                if file_id.endswith("txt"):
                    continue
                if file_id.endswith("xlsx"):
                    file_delete = os.path.join(root_dir, file_id.decode("UTF-8"))
                    os.remove("\"" + file_delete + "\"")

                if file_id not in image_list:
                    file_delete = os.path.join(root_dir, file_id)
                    current_app.logger.info("package:{}, file wrong:[{}]".format(package_index, file_id))
                    if os.path.exists(file_delete):
                        if os.path.isfile(file_delete):
                            os.remove(file_delete)
        else:
            for dir in self.dir_path:
                package_index = os.path.basename(dir)
                lis = os.listdir(dir)
                for i in lis:
                    i = str(i).strip()
                    if i.startswith('ext-') and i.endswith('png'):
                        continue
                    if i.startswith('label-') and i.endswith('png'):
                        continue
                    if i.startswith('ext-') and i.endswith('jpg'):
                        name = i[4:-11]
                        image_path = os.path.join(dir, i)
                        image = i
                        label = 'ext-label-' + name + '_00_004.png'
                        label_path = os.path.join(dir, label)
                    if i[0].isdigit() and i.endswith('jpg'):
                        image_path = os.path.join(dir, i)
                        name = i[:-11]
                        image = i
                        label = 'label-' + name + '_00_004.png'
                        label_path = os.path.join(dir, label)

                    if os.path.exists(image_path) and os.path.exists(label_path):
                        image_list.append(image)
                        image_list.append(label)

                        if image.startswith("ext-"):
                            origin_image = image[4:]
                            image_list.append(str(origin_image).strip())

                            annotation_file = origin_image[:-3] + "png"
                            image_list.append(str(annotation_file).strip())

                        if label.startswith("ext-"):
                            origin_label = label[4:]
                            image_list.append(str(origin_label).strip())

                    if not os.path.exists(image_path):
                        if os.path.exists(label_path):
                            current_app.logger.info("package:{}, not match:[{}] not exist=>[{}] exist".format(
                                package_index, image, label))
                            if os.path.exists(label_path):
                                os.remove(label_path)
                        else:
                            current_app.logger.info("package:{}, file missing:[{}]=>[{}]".format(
                                package_index, image, label))
                    else:
                        if not os.path.exists(label_path):
                            current_app.logger.info("package:{}, not match:[{}] exist=>[{}] not exist".format(
                                package_index, image, label))
                            if os.path.exists(image_path):
                                os.remove(image_path)
                    if i.endswith("txt"):
                        continue
                    if i.endswith("xlsx"):
                        file_delete = os.path.join(dir, i.decode("UTF-8"))
                        os.remove("\"" + file_delete + "\"")

                    if i not in image_list:
                        file_delete = os.path.join(dir, i)
                        current_app.logger.info("package:{}, file wrong:[{}]".format(package_index, i))
                        if os.path.exists(file_delete):
                            if os.path.isfile(file_delete):
                                os.remove(file_delete)

    def update_task(self):
        while True:
            plan = (1 - (int(self.remote_process_count.value) / (int(self.total_count) * 1.00))) * 100
            if plan < 100:
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                value = self.callback(self.task_id, self.status)
            if value:
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(5)
