# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.base_select import BaseSelectTask
from Apps.modules.base_train import BaseTrainTask
from Apps.modules.auto_select_task import AutoSelectTask, AutoSamTask
import uuid
import multiprocessing
import sys
import os
import re


reload(sys)
sys.setdefaultencoding('utf-8')


# 随机抽图
@api.route('/auto_sam', methods=["POST"])
def auto_sams():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseSelectTask()
    if task_id:
        _task.task_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        data = _task.task_history()
        resp = jsonify(errno=_task.error_code, data=data)
    return resp


@api.route('/sam_value', methods=["POST"])
def auto_sam_value():
    req_dict = request.json
    output_dir = req_dict.get('output_dir')
    sele_ratio = req_dict.get('ratio')
    track_file = req_dict.get('track_file')
    task_file = req_dict.get('task_file')
    isshuffle = req_dict.get('isshuffle')
    gpus = req_dict.get('gpus')
    weights_dir = req_dict.get('weights_dir')
    task_dir = os.path.join('/data/deeplearning/train_platform/select_data/out_image', output_dir)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    files_li = os.listdir(task_dir)
    li = []
    for i in files_li:
        if not re.match(r'.*sele.*', i):
            li.append(int(i[-1]))
    if li:
        li.sort()
        output_dir = os.path.join(task_dir, output_dir + str(sele_ratio) + '.v' + str(li[-1] + 1))
    else:
        output_dir = os.path.join(task_dir, output_dir + str(sele_ratio) + '.v1')
    task_type = 'AutoSamTask'
    status = 'starting'
    task_id = uuid.uuid1()
    auto_sam = AutoSamTask()
    pro = multiprocessing.Process(
        target=auto_sam.start,
        args=(output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status, isshuffle, task_type, task_file))
    pro.start()
    resp = jsonify(errno=auto_sam.error_code, message=auto_sam.message, data={"task_id": task_id})
    return resp


# 自动挑图
@api.route('/auto_sele', methods=["POST"])
def auto_sele():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseSelectTask()
    if task_id:
        _task.task_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        data = _task.task_history()
        resp = jsonify(errno=_task.error_code, data=data)
    return resp


@api.route('/sele_value', methods=["POST"])
def auto_sele_value():
    req_dict = request.json
    output_dir = req_dict.get('output_dir')
    sele_ratio = req_dict.get('ratio')
    track_file = req_dict.get('track_file')
    task_file = req_dict.get('task_file')
    isshuffle = req_dict.get('isshuffle')
    gpus = req_dict.get('gpus')
    weights_dir = req_dict.get('weights_dir')
    task_dir = os.path.join('/data/deeplearning/train_platform/select_data/out_image', output_dir)
    if not os.path.exists(task_dir):
        os.makedirs(task_dir)
    output_dir = os.path.join(task_dir, output_dir + '.sele')
    task_type = 'AutoSeleTask'
    status = 'starting'
    task_id = uuid.uuid1()
    auto_select = AutoSelectTask()
    pro = multiprocessing.Process(
        target=auto_select.start,
        args=(output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status, isshuffle, task_type, task_file))
    pro.start()
    resp = jsonify(errno=auto_select.error_code, message=auto_select.message, data={"task_id": task_id})
    return resp


# 自动挑图中模型选择
@api.route('/model_lists', methods=["GET"])
def model_lists():
    li = []
    _task = BaseTrainTask()
    _task.task_history()
    task_list = _task.process
    for task in task_list:
        li.append(task['task_name'])
    resp = jsonify(errno=1, data=li)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
