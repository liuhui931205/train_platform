# -*-coding:utf-8-*-
from . import api
from flask import jsonify, request
from Apps.modules.base_check import BaseCkeckTask
import uuid
import sys
from Apps.modules.near_farm_check import CheckTasks

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/check_data', methods=["POST"])
def check_data():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseCkeckTask()
    if task_id:
        _task.task_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        data = _task.task_history()
        resp = jsonify(errno=_task.error_code, data=data)
    return resp


@api.route('/check_sele', methods=["POST"])
def check_sele():
    req_dict = request.json
    task_name = req_dict.get('task_name')
    trackpointids = req_dict.get('trackpointids')
    weights_dir = req_dict.get('weights_dir')
    gpus = req_dict.get('gpus')
    status = 'starting'
    task_id = uuid.uuid1()
    check = CheckTasks()
    check.starts(task_name, trackpointids, gpus, task_id, weights_dir, status)
    resp = jsonify(errno=check.error_code, message=check.message, data={"task_id": task_id})
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response