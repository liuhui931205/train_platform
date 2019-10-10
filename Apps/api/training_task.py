# -*- coding:utf-8 -*-
import os
import json
from . import api
from flask import jsonify, request, current_app
from Apps.models import TrainTask
from Apps.modules.base_train import BaseTrainTask
from Apps.modules.train_task import TrainTasks, Datas
from Apps.modules.base_data import BaseDataTask
import uuid
from Apps import db, mach_id
import requests
import sys
from config import url_mach, data0_sour_train, data_dest_data, data_sour_train
import shutil
from Apps.utils.client import client

reload(sys)
sys.setdefaultencoding('utf-8')


# 创建训练任务
@api.route("/create_train", methods=["POST"])
def create_train():
    req_dict = request.json
    task_name = req_dict.get('train_name')
    network = req_dict.get('network')
    data = req_dict.get('data')
    task_desc = req_dict.get('task_desc')
    taskid = req_dict.get('task_id')
    types = req_dict.get('types')
    image_type = req_dict.get('image_type')
    parallel_bool = req_dict.get("parallel_bool")
    map_template = req_dict.get("maps")
    host = req_dict.get("host")
    if parallel_bool and parallel_bool != "0":
        parallel_bool = "1"
        host = "163"
    else:
        parallel_bool = "0"
    if host != mach_id:
        url = url_mach[host]
        url = url + "create_train"
        data = {
            'train_name': task_name,
            "network": network,
            "data": data,
            "task_desc": task_desc,
            "host": host,
            "types": types,
            "task_id": taskid,
            "image_type": image_type,
            "parallel_bool": parallel_bool,
            "maps": map_template
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
        task_id = uuid.uuid1()
        status = 'starting'
        traintask = TrainTasks()
        if not map_template:
            map_template = "general"
        traintask.create_task(task_name, network, data, task_desc, task_id, status, taskid, types, image_type,
                              parallel_bool, map_template)
        resp = jsonify(errno=traintask.error_code, message=traintask.message, data={"task_id": task_id})
    return resp


# 选择任务类型
@api.route('/task_type', methods=['GET'])
def task_type():
    """
    semantic-segmentation:语义分割
    OCR：ocr
    classification-model:分类模型
    target-detection:目标检测
    :return:
    """
    type_li = ['classification-model', 'target-detection', 'semantic-segmentation', 'OCR']
    resp = jsonify(errno=1, data=type_li)
    return resp


# 选择network
@api.route('/network', methods=['GET'])
def network():
    network_li = []
    dirs = os.listdir('/data/deeplearning/train_platform/network')
    for dir in dirs:
        if os.path.isdir(os.path.join('/data/deeplearning/train_platform/network', dir)):
            network_li.append(dir)
    if network_li:
        resp = jsonify(errno=1, data=network_li)
        return resp
    else:
        resp = jsonify(errno=0, message='请先创建network')
        return resp


# 选择datas
@api.route('/sour_datas', methods=['GET'])
def sour_datas():
    li = []
    _data = BaseDataTask()
    _data.data_history()
    datas_li = _data.process
    for data in datas_li:
        li.append(data['data_name'])
    if li:
        resp = jsonify(errno=1, data=li)
        return resp
    else:
        resp = jsonify(errno=0, message='请先创建data')
        return resp


@api.route('/continue_task', methods=['POST'])
def continue_task():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    traintask = TrainTasks()
    traintask.con_task(task_id)
    resp = jsonify(errno=traintask.error_code, message=traintask.message)
    return resp


# 结束训练
@api.route('/endtask', methods=['POST'])
def endtask():
    req_dict = request.json
    end_date = req_dict.get('end_date')
    task_id = req_dict.get('task_id')
    status = 'end'
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    if host != mach_id:
        url = url_mach[host]
        url = url + "continue_task"
        data = {"task_id": task_id, "end_date": end_date}
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        if js_data['errno'] == 1:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
    else:
        traintask = TrainTasks()
        traintask.end_task(task_id, end_date, status)
        resp = jsonify(errno=traintask.error_code, message=traintask.message)
    return resp


# 开始训练
@api.route('/starttask', methods=['POST'])
def starttask():
    req_dict = request.json
    data_type = req_dict.get('data_type')
    category_num = req_dict.get('num_classes')
    gpus = req_dict.get('gpus')
    batch_size = req_dict.get('batch_size')
    weight = req_dict.get('weights')
    iter_num = req_dict.get('num_epoch')
    model = req_dict.get('model')
    steps = req_dict.get('steps')
    learning_rate = req_dict.get('base_lr')
    start_date = req_dict.get('start_date')
    train_con = req_dict.get('train_con')
    data_con = req_dict.get('data_con')
    task_id = req_dict.get('task_id')
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    # with open(os.path.join(data_sour_train, task_name, "output/models/map_label.json"), 'r') as f:
    #     jss = json.loads(f.read())
    # category_num = str(len(jss) - 2)
    if host != mach_id:
        url = url_mach[host]
        url = url + "starttask"
        data = {
            "task_id": task_id,
            "data_type": data_type,
            "num_classes": category_num,
            "gpus": gpus,
            "batch_size": batch_size,
            "weights": weight,
            "num_epoch": iter_num,
            "model": model,
            "steps": steps,
            "base_lr": learning_rate,
            "start_date": start_date,
            "train_con": train_con,
            "data_con": data_con
        }
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        if js_data['errno'] == 1:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
    else:
        status = 'starting'
        traintask = TrainTasks()
        traintask.start_task(task_id, data_type, category_num, gpus, batch_size, weight, iter_num, model, steps,
                             learning_rate, start_date, train_con, data_con, status)
        resp = jsonify(errno=traintask.error_code, message=traintask.message)
    return resp


# 计算
@api.route('/calculate', methods=['GET'])
def calculate():
    task_id = request.args.get('task_id')
    batch_val = request.args.get('batch_val')
    try:
        traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        host = traintask.machine_id
        if host != mach_id:
            url = url_mach[host]
            url = url + "calculate?task_id={}&batch_val={}".format(task_id, batch_val)
            json_datas = requests.get(url=url)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'], data=js_data['data'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            batch_val = int(batch_val)
            data_path = traintask.data_path
            value = os.popen('cat ' + data_path + '/kd_all_train.lst  | wc -l').readlines()[0]
            value = int(value.strip())
            a = value / batch_val
            b = int(10000 / a)
            if b > 300:
                re_val = b
            else:
                re_val = 300
            resp = jsonify(errno=1, message='success', data=re_val)
    except Exception as e:
        current_app.logger.error(e)
        resp = jsonify(errno=0, message='no task id')
    return resp


# @api.route('/label_class', methods=['GET'])
# def label_class():
#     task_id = request.args.get('task_id')
#     try:
#         traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
#         host = traintask.machine_id
#         task_name = traintask.task_name
#         if host != mach_id:
#             url = url_mach[host]
#             url = url + "label_class?task_id={}".format(task_id)
#             json_datas = requests.get(url=url)
#             js_data = json.loads(json_datas.text)
#             if js_data['errno'] == 1:
#                 resp = jsonify(errno=js_data['errno'], message=js_data['message'], data=js_data['data'])
#             else:
#                 resp = jsonify(errno=js_data['errno'], message=js_data['message'])
#         else:
#
#             with open(os.path.join(data_sour_train, task_name, "output/models/map_label.json"), 'r') as f:
#                 jss = json.loads(f.read())
#             re_val = len(jss) - 2
#             resp = jsonify(errno=1, message='success', data=re_val)
#     except Exception as e:
#         current_app.logger.error(e)
#         resp = jsonify(errno=0, message='no task id')
#     return resp


# 选择weights
@api.route('/weights', methods=['GET'])
def weights():
    task_id = request.args.get('task_id')
    try:
        traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        host = traintask.machine_id
        if host != mach_id:
            url = url_mach[host]
            url = url + "weights?task_id={}".format(task_id)
            json_datas = requests.get(url=url)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'], data=js_data['data'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            task_name = traintask.task_name
            li = []
            # weight = '/data/deeplearning/train_platform/train_task/{}/models'.format(task_name)
            weight = os.path.join(data_sour_train, task_name, "models")
            files = os.listdir(weight)
            for file in files:
                if file.endswith('.params'):
                    li.append(file)
            if li:
                resp = jsonify(errno=1, message='query success', data=li)
            else:
                resp = jsonify(errno=0, message='no model found')
    except Exception as e:
        current_app.logger.error(e)
        resp = jsonify(errno=0, message='no task id')
    return resp


# 查询修改训练配置
@api.route('/train_config', methods=["GET"])
def train_config():
    task_id = request.args.get('task_id')
    try:
        traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        host = traintask.machine_id
        status = traintask.status
        task_type = traintask.type
        data_id = traintask.data_desc_id
        data = db.session.query(Datas).filter_by(id=data_id).first()
        data_name = data.data_name

        if host != mach_id:
            url = url_mach[host]
            url = url + "train_config?task_id={}".format(task_id)
            json_datas = requests.get(url=url)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'], data=js_data['data'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
                task_name = traintask.task_name
                task_type = traintask.type
                parallel_bool = traintask.parallel_bool
                if task_type == "semantic-segmentation":
                    # con_path = '/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json'.format(task_name)
                    con_path = os.path.join(data_sour_train, task_name, "conf/seg_train.json")
                    with open(con_path, 'rb') as f:
                        resp_datas = json.load(f)
                        data_s = resp_datas["train"]["data_conf"] = os.path.join(data_sour_train, task_name,
                                                                                 "conf/seg_train_data.json")
                        data_t = resp_datas["train"]["weights"] = os.path.join(
                            data_sour_train, task_name, "models/cityscapes_rna-a1_cls19_s8_ep-0001.params")
                        data_v = resp_datas["_condidate"]["weights"] = os.path.join(
                            data_sour_train, task_name, "models/cityscapes_rna-a1_cls19_s8_ep-0001.params")
                        if int(parallel_bool):
                            data_v = resp_datas["_condidate"]["kvstore"] = 'dist_sync'
                            data_t = resp_datas["train"]["kvstore"] = 'dist_sync'
                            data_v = resp_datas["logging"]["loggers"]["multiprocessing"]["handlers"].remove("console")
                            data_v = resp_datas["logging"]["loggers"]["multiprocessing"]["handlers"].append("mp_console")
                            data_v = resp_datas["logging"]["root"]["handlers"].append("console")
                            # data_v = resp_datas["logging"]["handlers"]["mp_console"].append("console")
                            data_v = resp_datas["logging"]["handlers"]["mp_console"]["stream"] = "ext://sys.stdout"
                            data_v = resp_datas["logging"]["handlers"]["console"]["stream"] = "ext://sys.stdout"

                    with open(con_path, 'wb') as f:
                        json.dump(resp_datas, f)
                    with open(con_path, 'r') as f:
                        resp_data = f.read()
                    resp = jsonify(errno=1, message='query success', data=resp_data)
                elif task_type == "classification-model":
                    # con_path = '/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json'.format(task_name)
                    con_path = os.path.join(data_sour_train, task_name, "conf/seg_train.json")
                    with open(con_path, 'rb') as f:
                        resp_datas = json.load(f)
                        data_s = resp_datas["train"]["data_conf"] = os.path.join(data_sour_train, task_name,
                                                                                 "conf/seg_train_data.json")
                        data_t = resp_datas["train"]["weights"] = os.path.join(
                            data_sour_train, task_name, "models/cityscapes_rna-a1_cls19_s8_ep-0001.params")
                        data_v = resp_datas["_condidate"]["weights"] = os.path.join(
                            data_sour_train, task_name, "models/cityscapes_rna-a1_cls19_s8_ep-0001.params")
                    with open(con_path, 'wb') as f:
                        json.dump(resp_datas, f)
                    with open(con_path, 'r') as f:
                        resp_data = f.read()
                    resp = jsonify(errno=1, message='query success', data=resp_data)
                elif task_type == "target-detection":
                    con_path = os.path.join(data_sour_train, task_name, "conf/tar_train.json")
                    with open(con_path, 'rb') as f:
                        data_v = resp_data = json.load(f)
                        data_v = resp_data["rpn"]["path_imgrec"] = os.path.join(data_dest_data, data_name,
                                                                                "masks_datas_train.rec")
                        data_v = resp_data["rcnn"]["path_imgrec"] = os.path.join(
                            data_sour_train, task_name, "network/rpn_data_iter/kuandeng_train_rpn_info.rec")
                    with open(con_path, 'wb') as f:
                        json.dump(resp_data, f)
                    with open(con_path, 'r') as f:
                        resp_data = f.read()
                    resp = jsonify(errno=1, message='query success', data=resp_data)
    except Exception as e:
        current_app.logger.error(e)
        resp = jsonify(errno=0, message='query fail')
    finally:
        return resp


# 查询修改数据配置
@api.route('/data_config', methods=["GET"])
def data_config():
    task_id = request.args.get('task_id')
    try:
        traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        host = traintask.machine_id
        if host != mach_id:
            url = url_mach[host]
            url = url + "data_config?task_id={}".format(task_id)
            json_datas = requests.get(url=url)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'], data=js_data['data'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            task_name = traintask.task_name
            task_type = traintask.type
            data_id = traintask.data_desc_id
            data = db.session.query(Datas).filter_by(id=data_id).first()
            data_name = data.data_name
            data_type = data.data_type
            if task_type == "semantic-segmentation":
                con_path = os.path.join(data_sour_train, task_name, "conf/seg_train_data.json")
                with open(con_path, 'rb') as f:
                    resp_datas = json.load(f)
                    data_s = resp_datas["common"]["label_map_file"] = os.path.join(data_sour_train, task_name,
                                                                                   "conf/label_map.txt")
                    data_t = resp_datas["train"]["path_imgrec"] = os.path.join(data_dest_data, data_name,
                                                                               "kd_{}_train.rec".format(data_type))
                    data_v = resp_datas["val"]["path_imgrec"] = os.path.join(data_dest_data, data_name,
                                                                             "kd_{}_val.rec".format(data_type))
                with open(con_path, 'wb') as f:
                    json.dump(resp_datas, f)
                with open(con_path, 'r') as f:
                    resp_data = f.read()
                resp = jsonify(errno=1, message='query success', data=resp_data)
            elif task_type == "classification-model":
                # con_path = '/data/deeplearning/train_platform/train_task/{}/conf/seg_train_data.json'.format(task_name)
                con_path = os.path.join(data_sour_train, task_name, "conf/seg_train_data.json")
                with open(con_path, 'rb') as f:
                    resp_datas = json.load(f)
                    data_s = resp_datas["common"]["label_map_file"] = os.path.join(data_sour_train, task_name,
                                                                                   "conf/label_map.txt")
                    data_t = resp_datas["train"]["path_imgrec"] = os.path.join(data_dest_data, data_name,
                                                                               "kd_{}_train.rec".format(data_type))
                    data_v = resp_datas["val"]["path_imgrec"] = os.path.join(data_dest_data, data_name,
                                                                             "kd_{}_val.rec".format(data_type))
                with open(con_path, 'wb') as f:
                    json.dump(resp_datas, f)
                with open(con_path, 'r') as f:
                    resp_data = f.read()
                resp = jsonify(errno=1, message='query success', data=resp_data)
            elif task_type == "target-detection":
                con_path = os.path.join(data_sour_train, task_name, "conf/tar_train.json")
                with open(con_path, 'r') as f:
                    resp_data = json.load(f)
                    resp_data["rpn"]["path_imgrec"] = os.path.join(data_dest_data, data_name, "masks_datas_train.rec")
                    resp_data["rcnn"]["path_imgrec"] = os.path.join(
                        data_sour_train, task_name, "network/rpn_data_iter/kuandeng_train_rpn_info.rec")
                with open(con_path, 'wb') as f:
                    json.dump(resp_data, f)
                with open(con_path, 'r') as f:
                    resp_data = f.read()
                resp = jsonify(errno=1, message='query success', data=resp_data)
    except Exception as e:
        current_app.logger.error(e)
        resp = jsonify(errno=0, message='query fail')
    finally:
        return resp


# 查询历史任务
@api.route('/train_history', methods=['POST'])
def train_history():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    task = BaseTrainTask()
    if task_id:
        task_name, task_type, host = task.task_status(task_id)
    else:
        task.task_history()
    resp = jsonify(errno=task.error_code, data=task.process)
    return resp


# 查询训练配置
@api.route('/train_configs', methods=["GET"])
def train_configs():
    task_id = request.args.get('task_id')
    traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
    task_type = traintask.type
    task_name = traintask.task_name
    status = traintask.status
    if task_type == "target-detection":
        # con_path = '/data/deeplearning/train_platform/train_task/{}/conf/tar_train.json'.format(task_name)
        con_path = os.path.join(data_sour_train, task_name, "conf/tar_train.json")
        sour_path = os.path.join(data0_sour_train, task_name, "conf/tar_train.json")
    else:
        # con_path = '/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json'.format(task_name)
        con_path = os.path.join(data_sour_train, task_name, "conf/seg_train.json")
        sour_path = os.path.join(data0_sour_train, task_name, "conf/seg_train.json")
    # get_path = '/data/deeplearning/train_platform/train_task/{}/conf'.format(task_name)
    get_path = os.path.join(data_sour_train, task_name, "conf")
    if not os.path.exists(get_path):
        os.makedirs(get_path)
    if status == "completed!":
        dest_scp, dest_sftp, dest_ssh = client(id="host")
        if not os.path.exists(con_path):
            dest_sftp.get(sour_path, con_path)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
    with open(con_path, 'r') as f:
        resp_data = f.read()
    resp = jsonify(errno=1, data=resp_data)
    return resp


# 查询数据配置
@api.route('/data_configs', methods=["GET"])
def data_configs():
    task_id = request.args.get('task_id')

    traintask = db.session.query(TrainTask).filter_by(task_id=task_id).first()
    task_type = traintask.type
    task_name = traintask.task_name
    status = traintask.status
    if task_type == "target-detection":
        con_path = os.path.join(data_sour_train, task_name, "conf/tar_train.json")
        sour_path = os.path.join(data0_sour_train, task_name, "conf/tar_train.json")
    else:
        con_path = os.path.join(data_sour_train, task_name, "conf/seg_train_data.json")
        sour_path = os.path.join(data0_sour_train, task_name, "conf/seg_train_data.json")
    get_path = os.path.join(data_sour_train, task_name, "conf")
    if not os.path.exists(get_path):
        os.makedirs(get_path)
    if status == "completed!":
        dest_scp, dest_sftp, dest_ssh = client(id="host")
        if not os.path.exists(con_path):
            dest_sftp.get(sour_path, con_path)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
    with open(con_path, 'r') as f:
        resp_data = f.read()
    resp = jsonify(errno=1, data=resp_data)

    return resp


# 查询映射模板
@api.route('/map_template', methods=["GET"])
def map_template():
    resp_data = ['mobilenet', 'segmentation-full', 'segmentation-half', "general"]
    resp = jsonify(errno=1, data=resp_data)
    return resp


@api.route('/map_info', methods=["GET"])
def map_info():
    task_id = request.args.get('task_id')
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    con_path = os.path.join(data_sour_train, task_name, "conf/label_map.txt")
    sour_con_path = os.path.join(data0_sour_train, task_name, "conf/label_map.txt")
    con_pathx = os.path.join(data_sour_train, task_name, "output/models/map_label.json")
    sour_con_pathx = os.path.join(data0_sour_train, task_name, "output/models/map_label.json")
    get_path = os.path.join(data_sour_train, task_name, "conf")
    get_pathx = os.path.join(data_sour_train, task_name, "output/models")
    dest_scp, dest_sftp, dest_ssh = client(id="host")

    if not os.path.exists(get_path):
        os.makedirs(get_path)
    if not os.path.exists(get_pathx):
        os.makedirs(get_pathx)
    if not os.path.exists(con_path):
        dest_sftp.get(sour_con_path, con_path)
    if not os.path.exists(con_pathx):
        dest_sftp.get(sour_con_pathx, con_pathx)
    li = []
    with open(con_pathx, 'r') as f:
        map_json = json.loads(f.read())
    with open(con_path, 'r') as f:
        while True:
            line = f.readline()
            if line:
                line = line.split('\t')

                for j in map_json:
                    if line[1].strip() == j["categoryId"]:
                        color = ""
                        for c in j["color"]:
                            color += str(c) + ','

                        li.append({
                            'id': line[0],
                            'categoryId': line[1].strip(),
                            'en_name': line[2].strip(),
                            'name': line[3].strip(),
                            'color': color.strip(',')
                        })

            else:
                break
    li.pop(0)
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    resp = jsonify(errno=1, data=li)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
