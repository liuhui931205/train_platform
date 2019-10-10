# -*- coding:utf-8 -*-
from Apps.libs.GPUtil.GPUtil import getGPUs
from . import api
from flask import jsonify
import sys

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/gups_info', methods=["GET"])
def gups_info():
    gpu_list = getGPUs()
    dicts = {}
    for k, v in gpu_list.items():
        gpu_li = []
        for gpu in v:
            gpu_value = {
                "pc-name": k,
                "id": gpu.id,
                "isLock": False,
                "gpuUtil": gpu.load,
                "totalMemory": gpu.memoryTotal,
                "freeMemory": gpu.memoryFree,
                "memoryutil": gpu.memoryUtil
            }
            gpu_li.append(gpu_value)
        dicts[k] = gpu_li
    resp = jsonify(errno=1, data=dicts)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
