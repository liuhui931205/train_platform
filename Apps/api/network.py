# -*- coding:utf-8 -*-
from . import api
from flask import request, jsonify
import os
from Apps.modules.net_task import NetTasks
from Apps.modules.base_network import BaseNetTask
import uuid
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


# 创建network
@api.route("/create_net", methods=["POST"])
def create_network():
    req_dict = request.json
    net_describe = req_dict.get('net_describe')
    net_name = req_dict.get('network_name')
    src_network = req_dict.get('src_net')
    task_id = uuid.uuid1()
    status = 'starting'
    nettask = NetTasks()
    nettask.create_network(task_id, net_name, src_network, net_describe, status)
    resp = jsonify(errno=nettask.error_code, message=nettask.message, data={"task_id": task_id})
    return resp


# 选择network源文件
@api.route('/src_network', methods=['GET'])
def src_network():
    src_network_li = []
    dirs = os.listdir('/data/deeplearning/network_template')
    for dir in dirs:
        if os.path.isdir(os.path.join('/data/deeplearning/network_template', dir)):
            src_network_li.append(dir)
    if src_network_li:
        resp = jsonify(errno=1, data=src_network_li)
        return resp
    else:
        resp = jsonify(errno=0, message='no found sour file')
        return resp


# 查询network历史
@api.route('/network_history', methods=['GET'])
def network_history():
    task_id = request.args.get('task_id')
    _net = BaseNetTask()
    if task_id:
        _net.net_status(task_id)
        resp = jsonify(errno=_net.error_code, data=_net.process)
    else:
        data = _net.net_history()
        resp = jsonify(errno=_net.error_code, data=data)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
