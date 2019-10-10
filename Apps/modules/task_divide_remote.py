# -*- coding:utf-8 -*-
import os
import time
import multiprocessing
import random
import cv2
import numpy as np
from flask import current_app
from Apps.utils.utils import get_file
from Apps.utils.utils import RemoteTask
from config import GlobalVar
from Apps.modules.base_divide import BaseDivideTask
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client


class TaskDivideRemoteHandler(BaseDivideTask):
    def __init__(self):
        super(BaseDivideTask, self).__init__()
        self.file_list = list()
        self.pixel = 50
        self.step = 20
        self.start = 1
        self.task_id = ''
        self.status = ''
        self.error_code = 1
        self.total = None
        self.message = 'Task Remote Start'
        self.src_dir = GlobalVar.image_dir.value
        manager = multiprocessing.Manager()
        self.remote_extend_queue = manager.Queue()
        self.remote_extend_count = manager.Value("i", 0)
        if not os.path.exists(self.src_dir):
            os.makedirs(self.src_dir)
        self.dest_dir = GlobalVar.package_dir.value
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def start_divide(self, version, step, types, task_id, status):
        self.task_id = task_id
        pro = multiprocessing.Process(target=self.get, args=(version, step))
        self.starts(version, step, types, task_id, status)
        pro.start()
        self.error_code = 1
        self.message = 'success'

    def get(self, version, step):

        try:
            _ver = version
            step = int(step)
            self.step = step
            self.src_dir = os.path.join(self.src_dir, _ver)
            if (not os.path.exists(self.src_dir)) or (_ver == ""):
                return
            else:
                dir_list = os.listdir(self.dest_dir)
                cur_max = 0
                for dir in dir_list:
                    dirs_list = os.listdir(os.path.join(self.dest_dir, dir))
                    for dirs in dirs_list:
                        if os.path.isdir(os.path.join(self.dest_dir, dir, dirs)):
                            if dirs.isdigit():
                                if int(dirs) > cur_max:
                                    cur_max = int(dirs)
                if self.start <= cur_max:
                    self.start = cur_max + 1
                self.dest_dir = os.path.join(self.dest_dir, _ver)
                if not os.path.exists(self.dest_dir):
                    os.makedirs(self.dest_dir)
                self.prepare_task(
                    src_dir=self.src_dir,
                    dest_dir=self.dest_dir,
                    start_package=self.start,
                    cnt_per_package=self.step
                )
                _task = RemoteTask(
                    package_index=None,
                    src_path=None,
                    dest_path=None,
                    exit_flag=True
                )
                self.remote_extend_queue.put(_task)
                pro = multiprocessing.Process(target=self.update_task)
                pro.start()
                process = multiprocessing.Process(target=self.do)
                process.start()
                process.join()
                # copyFiles(self.dest_dir,os.path.join("/data0/dataset/training/data/packages",_ver))

                dest_scp, dest_sftp, dest_ssh = client(id="host")
                try:
                    dest_sftp.stat(self.dest_dir)
                    dest_ssh.exec_command("rm -r {}".format(self.dest_dir))
                except IOError:
                    pass
                    # dest_sftp.mkdir(self.dest_dir)
                dest_scp.put(self.dest_dir, self.dest_dir, recursive=True)
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
                pro.join()
        except Exception as err:
            print(err)

    def prepare_task(self, src_dir, dest_dir, start_package, cnt_per_package):
        # 生成标注任务包
        src_len = len(src_dir)
        get_file(src_dir, self.file_list, src_len)
        random.shuffle(self.file_list)
        total_count = len(self.file_list)
        self.total = total_count
        file_index = 0
        total_index = 0
        package_index = start_package
        package_list = {}
        for _file_path in self.file_list:
            total_index += 1

            _file_id = os.path.basename(_file_path)

            package_dir = os.path.join(dest_dir, str(package_index))
            if not os.path.exists(package_dir):
                os.makedirs(package_dir)

            image_file = _file_id

            _file_name = _file_id.split(".")
            _file_name = ".".join(_file_name[:-1])
            label_file = "label-" + _file_name + ".png"
            package_list[image_file] = label_file

            src_path = os.path.join(src_dir, _file_path)

            _image = cv2.imread(src_path)
            if _image is None:
                print(src_path)
                continue

            ext_image_name = "ext-" + _file_id
            dest_path = os.path.join(package_dir, ext_image_name)

            _task = RemoteTask(
                package_index=str(package_index),
                src_path=src_path,
                dest_path=dest_path
            )
            self.remote_extend_queue.put(_task)
            self.remote_extend_count.value += 1

            file_index += 1
            if file_index == cnt_per_package:
                dest_file = "ext-" + str(package_index) + ".csv"
                dest_file_path = os.path.join(dest_dir, str(package_index), dest_file)

                with open(dest_file_path, "w") as f:
                    for _image, _label in package_list.items():
                        _str = "ext-{},ext-{}\n".format(_image, _label)
                        f.write(_str)

                package_list = {}
                file_index = 0
                package_index += 1
            elif total_index == total_count:
                dest_file = "ext-" + str(package_index) + ".csv"
                dest_file_path = os.path.join(dest_dir, str(package_index), dest_file)

                with open(dest_file_path, "w") as f:
                    for _image, _label in package_list.items():
                        _str = "ext-{},ext-{}\n".format(_image, _label)
                        f.write(_str)
        return

    def do(self):
        if self.remote_extend_queue.empty():
            return

        while not self.remote_extend_queue.empty():
            _task = self.remote_extend_queue.get()

            if not isinstance(_task, RemoteTask):
                break

            if _task.exit_flag:
                break

            self.remote_extend_count.value -= 1

            src_path = _task.src_path
            dest_path = _task.dest_path

            _image = cv2.imread(src_path)
            if _image is None:
                print(src_path)
            else:
                width = _image.shape[1]
                height = _image.shape[0]
                crop_img = _image[int(height // 2):height, 0:width]  # 切图
                new_w = width + self.pixel * 2
                new_h = int(height // 2) + self.pixel * 2

                blank_image = np.zeros((new_h, new_w, 3), np.uint8)
                blank_image[0:new_h, 0:new_w] = (255, 255, 255)
                blank_image[self.pixel:new_h - self.pixel, self.pixel:new_w - self.pixel] = crop_img
                cv2.imwrite(dest_path, blank_image)

    def update_task(self):
        while True:
            plan = (1.00 - (int(self.remote_extend_count.value) / (int(self.total) * 1.00))) * 100
            if plan != 100:
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                value = self.callback(self.task_id, self.status)
            if value:
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(5)
