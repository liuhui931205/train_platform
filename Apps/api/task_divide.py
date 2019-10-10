# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.task_divide_full import TaskDivideFullHandler
from Apps.modules.task_divide_remote import TaskDivideRemoteHandler
from Apps.modules.base_divide import BaseDivideTask
from Apps.libs.target_divide import target_task
import uuid
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client
import os
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/divide_history', methods=['POST'])
def divide_history():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseDivideTask()
    if task_id:
        _task.divide_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        _task.divide_history()
        resp = jsonify(errno=_task.error_code, data=_task.process)
    return resp


@api.route('/task_divide', methods=["POST"])
def task_divide():
    req_dicts = request.json
    version = req_dicts.get('version')
    step = req_dicts.get('step')
    types = req_dicts.get('types')
    task_id = uuid.uuid1()
    status = 'starting'
    resp = None
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    path = "/data/deeplearning/dataset/training/data/images"
    # data0_sour = "/data0/dataset/training/data/images"
    dirs = dest_sftp.listdir(path)
    if version not in dirs:
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        resp = jsonify(errno=0, message="no found data")
    else:
        if not os.path.exists(os.path.join(path, version)):
            dest_scp.get(os.path.join(path, version), os.path.join(path, version), recursive=True)
            # dest_sftp.put(os.path.join(dir_name, i), os.path.join(dir_name, i))
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()

    if types == 'remote':
        task_remote = TaskDivideRemoteHandler()
        task_remote.start_divide(version, step, types, task_id, status)
        resp = jsonify(errno=task_remote.error_code, message=task_remote.message, data={"task_id": task_id})
    elif types == 'full':
        task_full = TaskDivideFullHandler()
        task_full.start_divide(version, step, types, task_id, status)
        resp = jsonify(errno=task_full.error_code, message=task_full.message, data={"task_id": task_id})
    return resp


@api.route('/target_divide', methods=["POST"])
def target_divide():
    req_dicts = request.json
    version = req_dicts.get('version')
    step = req_dicts.get('step')
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    path = "/data/deeplearning/dataset/training/data/images"
    # data0_sour = "/data0/dataset/training/data/images"
    # dirs = os.listdir(data0_sour)
    dirs = dest_sftp.listdir(path)
    if version not in dirs:
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        resp = jsonify(errno=0, message="no found data")
    else:
        # copyFiles(os.path.join(data0_sour, version), os.path.join(path, version))
        # dest_sftp.put(os.path.join(dir_name, i), os.path.join(dir_name, i))
        dest_scp.get(os.path.join(path, version), os.path.join(path, version), recursive=True)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        if not os.path.exists(os.path.join("/data/deeplearning/dataset/training/data/images", version)):
            os.makedirs(os.path.join("/data/deeplearning/dataset/training/data/images", version))
        if not os.path.exists(os.path.join("/data/deeplearning/dataset/training/data/packages", version)):
            os.makedirs(os.path.join("/data/deeplearning/dataset/training/data/packages", version))
        target_task(os.path.join("/data/deeplearning/dataset/training/data/images", version),
                    os.path.join("/data/deeplearning/dataset/training/data/packages", version), step)
        resp = jsonify(errno=1, message="success")
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
