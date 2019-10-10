# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.process_label_union import ProcessLabelUnionHandler
from Apps.modules.process_label_lane import ProcessLabelLaneHandler
from Apps.modules.process_label_remote import ProcessLabelRemoteHandler
from Apps.modules.base_process_label import BaseProcessLabel
from Apps.utils.label_process import rm_file
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client
import uuid
import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/label_history', methods=['POST'])
def label_history():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseProcessLabel()
    if task_id:
        _task.label_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        _task.label_history()
        resp = jsonify(errno=_task.error_code, data=_task.process)
    return resp


@api.route('/process_label', methods=["POST"])
def process_label():
    req_dicts = request.json
    version = req_dicts.get('version')
    types = req_dicts.get('types')
    color_info = req_dicts.get('color_info')
    task_id = uuid.uuid1()
    status = 'starting'
    resp = None
    path = "/data/deeplearning/dataset/training/data/released"
    data0_sour = "/data0/dataset/training/data/released"

    dest_scp, dest_sftp, dest_ssh = client(id="host")
    dirs = dest_sftp.listdir(path)
    if version not in dirs:
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        resp = jsonify(errno=0, message="no found data")
    else:
        dest_scp.get(os.path.join(path, version), os.path.join(path, version), recursive=True)
        # dest_sftp.put(os.path.join(dir_name, i), os.path.join(dir_name, i))
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        rm_file(version)
        if types == 'remote':
            task_remote = ProcessLabelRemoteHandler()
            task_remote.start_process(version, types, task_id, status, color_info)
            resp = jsonify(errno=task_remote.error_code, message=task_remote.message, data={"task_id": task_id})
        elif types == 'lane':
            task_full = ProcessLabelLaneHandler()
            task_full.start_process(version, types, task_id, status, color_info)
            resp = jsonify(errno=task_full.error_code, message=task_full.message, data={"task_id": task_id})
        elif types == 'union':
            task_full = ProcessLabelUnionHandler()
            task_full.start_process(version, types, task_id, status, color_info)
            resp = jsonify(errno=task_full.error_code, message=task_full.message, data={"task_id": task_id})
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
