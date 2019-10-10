# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.base_offline import BaseOfflineTask
from Apps.modules.offline_imp import OffTasks
import uuid
import sys
import os

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/off_status', methods=['POST'])
def off_status():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseOfflineTask()
    if task_id:
        _task.off_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        _task.off_history()
        resp = jsonify(errno=_task.error_code, data=_task.process)
    return resp


@api.route('/offimport', methods=['POST'])
def offimport():
    req_dicts = request.json
    roadelement = req_dicts.get('roadelement')
    source = req_dicts.get('source')
    author = req_dicts.get('author')
    annotype = req_dicts.get('annotype')
    datakind = req_dicts.get('datakind')
    city = req_dicts.get('city')
    name = req_dicts.get('src')
    imgoprange = req_dicts.get('imgoprange')
    src = '/data/deeplearning/train_platform/offline_input/{}'.format(name)
    dest = '/data/deeplearning/train_platform/offline_output/{}'.format(name)
    if not os.path.exists(src):
        os.makedirs(src)
    if not os.path.exists(dest):
        os.makedirs(dest)
    task_id = uuid.uuid1()
    status = 'starting'
    offtask = OffTasks()
    offtask.create(src, dest, task_id, roadelement, source, author, annotype, datakind, city, imgoprange, status)
    resp = jsonify(errno=offtask.error_code, message=offtask.message, data={"task_id": task_id})
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
