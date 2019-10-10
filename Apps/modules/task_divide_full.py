# -*- coding:utf-8 -*-
import os
import time
import multiprocessing
import random
import cv2
import shutil
from flask import current_app
from Apps.utils.utils import get_file
from config import GlobalVar
from Apps.modules.base_divide import BaseDivideTask
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client


class TaskDivideFullHandler(BaseDivideTask):
    def __init__(self):
        super(BaseDivideTask, self).__init__()
        self.file_list = list()
        self.pixel = 50
        self.step = 20
        self.start = 1
        self.task_id = ''
        self.status = ''
        self.src_dir = GlobalVar.image_dir.value
        self.error_code = 1
        self.message = 'Task Full Start'
        if not os.path.exists(self.src_dir):
            os.makedirs(self.src_dir)
        self.dest_dir = GlobalVar.package_dir.value + "_full"
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
        except Exception as err:
            print(err)

    def prepare_task(self, src_dir, dest_dir, start_package, cnt_per_package):
        # 生成标注任务包
        src_len = len(src_dir)
        get_file(src_dir, self.file_list, src_len)
        random.shuffle(self.file_list)
        total_count = len(self.file_list)
        file_index = 0
        total_index = 0
        package_index = start_package
        package_list = {}
        manager = multiprocessing.Manager()
        count = manager.Value('i', 0)
        cou_pro = multiprocessing.Process(target=self.update_task, args=(count, total_count))
        cou_pro.start()
        for _file_path in self.file_list:
            total_index += 1
            count.value = total_index
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
            file_index += 1
            if file_index == cnt_per_package:
                dest_file = "ext-" + str(package_index) + ".csv"
                dest_file_path = os.path.join(dest_dir, str(package_index), dest_file)
                with open(dest_file_path, "w") as f:
                    for _image, _label in package_list.items():
                        _str = "ext-{},ext-{}\n".format(_image, _label)
                        f.write(_str)
                        dest_path = os.path.join(package_dir, "ext-" + _image)
                        shutil.copy(os.path.join(src_dir, _image), dest_path)
                        # todo
                        tag_file = _image + ".csv"
                        if os.path.exists(os.path.join(src_dir, tag_file)):
                            shutil.copy(os.path.join(src_dir, tag_file), os.path.join(package_dir, tag_file))
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
                        dest_path = os.path.join(package_dir, "ext-" + _image)
                        shutil.copy(os.path.join(src_dir, _image), dest_path)
        cou_pro.join()

    def update_task(self, count, total_count):
        while True:
            if count.value != 0:
                plan = (int(count.value) / (int(total_count) * 1.00)) * 100
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
