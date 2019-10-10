# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
import os
from Apps.models import Datas
from Apps import db
from Apps.modules.base_data import BaseDataTask
from Apps.modules.data_task import DataTasks
import uuid
import multiprocessing
import sys
import json
from Apps.utils.copy_file import get_datas
from config import data0_sour_data,data0_dest_data,data_dest_data
import subprocess
from Apps.utils.client import client
from Apps import mach_id
import shutil
from stat import S_ISDIR

reload(sys)
sys.setdefaultencoding('utf-8')


# 选择data
@api.route('/data', methods=['GET'])
def data():
    li = []
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    dirs = dest_sftp.listdir(data0_sour_data)
    for dir in dirs:
        if S_ISDIR(dest_sftp.stat(os.path.join(data0_sour_data, dir)).st_mode):
            li.append(dir)
    li.sort()
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    resp = jsonify(errno=1, data=li)
    return resp


# 选择任务类型
@api.route('/dtask_type', methods=['GET'])
def dtask_type():
    type_li = ['classification-model', 'target-detection', 'semantic-segmentation', 'OCR']
    resp = jsonify(errno=1, data=type_li)
    return resp


# 生成Data
@api.route('/create_data', methods=['POST'])
def create_data():
    req_dict = request.json
    data_name = req_dict.get('data_name')
    data_type = req_dict.get('data_type')
    sour_data = req_dict.get('sour_data')
    types = req_dict.get('types')
    thread = req_dict.get('thread')
    image_type = req_dict.get('images_type')
    train = req_dict.get('train')
    val = req_dict.get('val')
    test = req_dict.get('test')
    data_describe = req_dict.get('data_desc')
    l_value = req_dict.get('l_value')
    status = 'starting'
    taskid = req_dict.get('task_id')
    task_id = uuid.uuid1()
    if not os.path.exists(os.path.join(data_dest_data, data_name)):
        os.makedirs(os.path.join(data_dest_data, data_name))
    get_datas(taskid, sour_data, data_name)
    manager = multiprocessing.Manager()
    datatask = DataTasks(manager=manager)
    datatask.create_data(taskid, task_id, data_name, data_type, train, val, test, sour_data, data_describe, status,
                         thread, l_value, types, image_type)
    resp = jsonify(errno=datatask.error_code, message=datatask.message, data={"task_id": task_id})
    return resp


# 查询Data历史及进度
@api.route('/data_history', methods=['POST'])
def create_rate():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _data = BaseDataTask()
    if task_id:
        _data.data_status(task_id)
        resp = jsonify(errno=_data.error_code, data=_data.process)
    else:
        _data.data_history()
        resp = jsonify(errno=_data.error_code, data=_data.process)
    return resp


@api.route('/class_count', methods=['GET'])
def class_count():
    task_id = request.args.get('task_id')
    if task_id:
        data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
        data_name = data_task.data_name
        images_type = data_task.images_type
        data_path = os.path.join(data0_dest_data, data_name)
        dest_path = "/data/deeplearning/train_platform/data_class_count"

        dest_scp, dest_sftp, dest_ssh = client(id="host")
        files = dest_sftp.listdir(data_path)
        if "class.json" in files:
            path = os.path.join(data_path, "class.json")
            dest_sftp.get(path, os.path.join(dest_path, "class.json"))
        else:
            cmd1 = "python2 /data/deeplearning/liuhui/train_platform/data_class_count/general_class.py {} {}".format(
                data_path, images_type)
            # p = subprocess.Popen(cmd1, shell=True, stdout=subprocess.PIPE)
            stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd1, timeout=100000)
            while True:
                info1 = stdout1.readline()
                err1 = stderr1.readline()
                if info1.strip() == "success":
                    break
            dest_sftp.get(os.path.join(data_path, "class.json"), os.path.join(dest_path, "class.json"))
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
        file_list = os.listdir(dest_path)
        for i in file_list:
            if i.endswith(".json"):
                with open(os.path.join(dest_path, i), 'r') as f:
                    data = json.loads(f.read())
                    data_list = []
                    for k, v in data.items():
                        v["id"] = int(k)
                        v["test"].pop()
                        v["val"].pop()
                        v["train"].pop()
                        data_list.append(v)
                resp = jsonify(errno=1, data=data_list)
    else:
        resp = jsonify(errno=0, message="failed")
    return resp


@api.route('/add_data', methods=['POST'])
def add_data():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    class_id = req_dict.get('class_id')
    add_list = []
    data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
    sour_data = data_task.sour_data
    types = data_task.type
    if types == "semantic-segmentation":
        # data_path = os.path.join("/data/deeplearning/train_platform/data", data_name)
        sour_data = os.path.join(data0_sour_data, sour_data)
        dest_path = "/data/deeplearning/train_platform/data_class_count"
        class_path = os.path.join(dest_path, "class.json")
        with open(class_path, 'r') as f:
            json_data = json.loads(f.read())
        for name in ["test", "train", "val"]:
            li = json_data[str(class_id)][name][-1]
            if isinstance(li, list):
                add_list.extend(li)
        add_dirs = []
        try:
            # if machine_id != mach_id:
            dest_scp, dest_sftp, dest_ssh = client(id="host")
            dirs = dest_sftp.listdir(sour_data)
            for d in dirs:
                if d.startswith("add_"):
                    add_dirs.append(d)
            if add_dirs:
                add_dirs.sort()
                dest_dir = os.path.join(sour_data, "add_" + str(int(add_dirs[-1].split("_")[-1]) + 1))
            else:
                dest_dir = os.path.join(sour_data, "add_1")
            dest_sftp.mkdir(dest_dir)
            for img in add_list:
                path = os.path.dirname(img)
                name = os.path.basename(img)
                cmd1 = "cp {} {}".format(img, dest_dir)
                stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd1, timeout=1000000)
                # os.system(cmd1)
                cmd2 = "cp {} {}".format(img[:-3] + "jpg", dest_dir)
                stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd2, timeout=100000)
                # os.system(cmd2)
                cmd3 = "cp {} {}".format(os.path.join(path, "label-" + name), dest_dir)
                # os.system(cmd3)
                stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd3, timeout=100000)
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            # else:
            #     dirs = os.listdir(sour_data)
            #     for d in dirs:
            #         if d.startswith("add_"):
            #             add_dirs.append(d)
            #     if add_dirs:
            #         add_dirs.sort()
            #         dest_dir = os.path.join(sour_data, "add_" + str(int(add_dirs[-1].split("_")[-1]) + 1))
            #     else:
            #         dest_dir = os.path.join(sour_data, "add_1")
            #     os.makedirs(dest_dir)
            #     for img in add_list:
            #         path = os.path.dirname(img)
            #         name = os.path.basename(img)
            #         shutil.copyfile(img, os.path.join(dest_dir, name))
            #         shutil.copyfile(img[:-3] + "jpg", os.path.join(dest_dir, name[:-3] + "jpg"))
            #         shutil.copyfile(os.path.join(path, "label-" + name), os.path.join(dest_dir, "label-" + name))
        except Exception as e:
            resp = jsonify(errno=0, message="failed")
        else:
            resp = jsonify(errno=1, message="success")
    else:
        resp = jsonify(errno=0, message="failed")
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
