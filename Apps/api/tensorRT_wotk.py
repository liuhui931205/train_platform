# -*-coding:utf-8-*-
from . import api
from flask import jsonify, request
from Apps.libs.tensor.ten_model import work
from Apps.modules.tens_work import TenTasks
from Apps import db, mach_id
from Apps.utils.client import client
from Apps.models import TrainTask, Datas
import requests
import json
import shutil
import os
from config import url_mach, data_sour_train, data0_sour_train
import uuid
from Apps.utils.copy_file import copy_trt_data
from Apps.utils.copy_all import copyFiles


@api.route('/tensor_w', methods=['POST'])
def tensor_w():
    req_dict = request.json
    dicts = req_dict.get('dicts')
    multiple4 = req_dict.get('multiple4')    # bool
    shape = req_dict.get('shape')
    shape = '\'' + shape + '\''
    gpu = req_dict.get('gpu')
    host = req_dict.get('server')
    maxBatchSize = req_dict.get('maxBatchSize')
    iname = req_dict.get('iname')
    oname = req_dict.get('oname')
    OIndex = req_dict.get('OIndex')
    topk = req_dict.get('topk')  # bool
    fp16 = req_dict.get('fp16')  # bool
    int8 = req_dict.get('int8')  # bool
    imagelstInt8 = req_dict.get('imagelstInt8')
    reuseCacheInt8 = req_dict.get('reuseCacheInt8')  # bool
    batchSizeInt8 = req_dict.get('batchSizeInt8')
    maxBatchesInt8 = req_dict.get('maxBatchesInt8')
    if host != mach_id:
        url = url_mach[host]
        url = url + "tensor_w"
        data = {
            'dicts': dicts,
            "multiple4": multiple4,
            "shape": shape,
            "gpu": gpu,
            "host": host,
            "maxBatchSize": maxBatchSize,
            "iname": iname,
            "oname": oname,
            "OIndex":OIndex,
            "topk":topk,
            "fp16":fp16,
            "int8":int8,
            "imagelstInt8":imagelstInt8,
            "reuseCacheInt8":reuseCacheInt8,
            "batchSizeInt8":batchSizeInt8,
            "maxBatchesInt8":maxBatchesInt8
        }
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        if js_data['errno'] == 1:
            resp = jsonify(errno=js_data['errno'],
                           message=js_data['message'],
                           data={"task_id": js_data['data']["task_id"]})
        else:
            resp = jsonify(errno=js_data['errno'],
                           message=js_data['message'],
                           data={"task_id": js_data['data']["task_id"]})

    else:

        if multiple4:
            multiple4 = 1
        else:
            multiple4 = 0
        if topk:
            topk = 1
        else:
            topk = 0
        if fp16:
            fp16 = 1
        else:
            fp16 = 0
        if int8:
            int8 = 1
        else:
            int8 = 0
        if reuseCacheInt8:
            reuseCacheInt8 = 1
        else:
            reuseCacheInt8 = 0

        if dicts:
            for k, v in dicts.items():
                if v:
                    train_task = db.session.query(TrainTask).filter_by(task_name=k).first()
                    if train_task:
                        machine_id = train_task.machine_id
                        status = train_task.status
                        get_path = os.path.join(data_sour_train,k,"output/models/")
                        if not os.path.exists(get_path):
                            os.makedirs(get_path)
                        if status != "completed!":
                            if machine_id != host:
                                dest_scp, dest_sftp, dest_ssh = client(id=machine_id)
                                for i in v:
                                    model = os.path.join(get_path, i)
                                    if not os.path.exists(model):
                                        dest_sftp.get(model, model)
                                mod_li = dest_sftp.listdir(get_path)
                                for a in mod_li:
                                    if a.endswith('.json') or a.endswith('.txt'):
                                        mp = os.path.join(get_path, a)
                                        if not os.path.exists(mp):
                                            dest_sftp.get(mp, mp)
                                dest_scp.close()
                                dest_sftp.close()
                                dest_ssh.close()
                        else:
                            dest_scp, dest_sftp, dest_ssh = client(id="host")
                            src_path = os.path.join(data0_sour_train,k,"output/models/")
                            for i in v:
                                model = os.path.join(get_path, i)
                                if not os.path.exists(model):
                                    dest_sftp.get(os.path.join(src_path,i), model)
                            mod_li = dest_sftp.listdir(src_path)
                            for a in mod_li:
                                if a.endswith('.json') or a.endswith('.txt'):
                                    mp = os.path.join(get_path, a)
                                    if not os.path.exists(mp):
                                        dest_sftp.get(os.path.join(src_path,a), mp)
                            dest_scp.close()
                            dest_sftp.close()
                            dest_ssh.close()



                            # copyFiles(src_path, get_path)
                        params = os.path.join(get_path, v[0])
                        js_li = os.listdir(get_path)
                        for j in js_li:
                            if j.endswith("symbol.json"):
                                network = os.path.join(get_path, j)
            info = work(
                params,
                network,
                gpu,
                multiple4=multiple4,
                shape=shape,
                maxBatchSize=maxBatchSize,
                iname=iname,
                oname=oname,
                OIndex=OIndex,
                topk=topk,
                fp16=fp16,
                int8=int8,
                imagelstInt8=imagelstInt8,
                reuseCacheInt8=reuseCacheInt8,
                batchSizeInt8=batchSizeInt8,
                maxBatchesInt8=maxBatchesInt8)
            if info == "success":
                if status != "completed!":
                    if machine_id != mach_id:
                        dest_scp, dest_sftp, dest_ssh = client(id=machine_id)
                        paths = os.path.dirname(params)
                        model_ll = os.listdir(paths)
                        files = dest_sftp.listdir(path=paths)
                        for i in model_ll:
                            if i not in files:
                                model = os.path.join(paths, i)
                                dest_sftp.put(model, model)
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()
                else:
                    dest_scp, dest_sftp, dest_ssh = client(id="host")
                    get_path = os.path.join(data_sour_train, k, "output/models/")
                    src_path = os.path.join(data0_sour_train, k, "output/models/")
                    model_ll = os.listdir(get_path)
                    files = dest_sftp.listdir(path=src_path)
                    for i in model_ll:
                        if i not in files:
                            model = os.path.join(get_path, i)
                            dest_sftp.put(model, os.path.join(src_path,i))
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
            resp = jsonify(errno=1, message=info, data={"task_id": "sadasdasdas"})
        else:
            resp = jsonify(errno=1, message="No choice", data={"task_id": "sadasdas"})
    return resp


@api.route('/tensor_eva', methods=['POST'])
def tensor_eva():
    req_dict = request.json
    dicts = req_dict.get('dicts')
    l_value = req_dict.get('l_value')
    type = req_dict.get('type')
    gpus = req_dict.get('gpus')
    single_gpu = req_dict.get('single_gpu')
    img = req_dict.get('img')
    argmax = req_dict.get('argmax')
    effective_c = req_dict.get("effective_c")
    task_id = uuid.uuid1()
    status = 'starting'
    sour_li, dest, models, task_type = copy_trt_data(l_value, dicts, mach_id, type)
    if all([sour_li, dest, models, task_type]):
        eva = TenTasks()
        eva.evaluating(task_id, sour_li, gpus, dest, single_gpu, models, status, task_type, img, argmax, effective_c)
        resp = jsonify(errno=eva.error_code, message=eva.message, data={"task_id": task_id})
    else:
        resp = jsonify(errno=0, message='no found model')
    return resp


@api.after_request
def af_request(response):
    """
    :param response:
    :return:
    """
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response