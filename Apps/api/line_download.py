# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.base_linedown import BaseLineDownTask
from Apps.modules.line_downs import LineTasks
from Apps.modules.track_down_data import track_ponit_down
import uuid
import sys
import os
import json
import requests
from config import json_path
from Apps import kds_meta

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/line_status', methods=['POST'])
def line_status():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    _task = BaseLineDownTask()
    if task_id:
        _task.line_status(task_id)
        resp = jsonify(errno=_task.error_code, data=_task.process)
    else:
        _task.line_history()
        resp = jsonify(errno=_task.error_code, data=_task.process)
    return resp


@api.route('/tag_linedown', methods=['POST'])
def linedown():
    req_dicts = request.json
    taskid_start = req_dicts.get('taskid_start')
    taskid_end = req_dicts.get('taskid_end')
    dest = req_dicts.get('dest')
    path = '/data/deeplearning/dataset/training/data/released'
    dest = os.path.join(path, dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
    task_id = uuid.uuid1()
    status = 'starting'
    linetask = LineTasks()
    linetask.create(task_id, taskid_start, taskid_end, dest, status)
    resp = jsonify(errno=linetask.error_code, message=linetask.message, data={"task_id": task_id})
    return resp


# todo 根据标签下载
@api.route('/linedown', methods=['POST'])
def tag_linedown():
    req_dicts = request.json
    taskid_start = req_dicts.get('taskid_start')
    taskid_end = req_dicts.get('taskid_end')
    dest = req_dicts.get('dest')

    path = '/data/deeplearning/dataset/training/data/released'
    dest = os.path.join(path, dest)
    if not os.path.exists(dest):
        os.makedirs(dest)
    task_id = uuid.uuid1()
    status = 'starting'
    linetask = LineTasks()
    linetask.create(task_id, taskid_start, taskid_end, dest, status)
    resp = jsonify(errno=linetask.error_code, message=linetask.message, data={"task_id": task_id})
    return resp


# 标签信息
@api.route('/tags_info', methods=['GET'])
def tags_info():
    js_datas = requests.get(url=kds_meta)
    info_data = []
    data = json.loads(js_datas.text)
    if data['code'] == '0' and data['message'] == u'成功':
        info_lis = data["result"]["layers"][0]["model"]["fields"]
        for i in info_lis:
            dicts = {}
            for k, v in i.items():
                if k == "fieldName":
                    dicts["fname"] = v
                elif k == "fieldTitle":
                    dicts["ftitle"] = v
                elif k == "fieldType":
                    li = []
                    if not v:
                        break
                    for j in v['fieldTypeValues']:
                        t_dict = {}
                        for q, e in j.items():
                            if q == "name":
                                t_dict["name"] = e
                            elif q == "value":
                                t_dict["value"] = e
                        li.append(t_dict)

                    dicts["ftype"] = li
            if len(dicts) == 3:
                info_data.append(dicts)
    resp = jsonify(errno=1, data=info_data)
    return resp


# 轨迹点下载
@api.route('/track_down', methods=['POST'])
def track_down():
    req_dicts = request.json
    dir_name = req_dicts.get('dir_name')
    trackpointids = req_dicts.get('trackpointids')
    trackpoint_list = trackpointids.split('\n')
    track_ponit_down(dir_name, trackpoint_list)
    resp = jsonify(errno=1, message="success")
    return resp


# json文件列表
@api.route('/json_files', methods=['GET'])
def json_files():
    data = []
    json_li = os.listdir(json_path)
    for i in json_li:
        if i.endswith("json"):
            data.append(i)
    resp = jsonify(errno=1, data=data)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
