# -*- coding:utf-8 -*-
from . import api
import os
from flask import jsonify, request, current_app
import base64
from Apps.modules.evaluate_task import EvaTasks
from Apps.modules.base_eva import BaseEvaTask
import uuid
import json
import sys
import requests
from Apps.utils.client import client
from config import url_mach
from Apps.utils.copy_file import copy_data
from Apps.models import Confidence_Datas
from Apps import db, redis_store, mach_id

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/eva_rate', methods=['POST'])
def eva_rate():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    task = BaseEvaTask()
    if task_id:
        task.eva_status(task_id)
    else:
        task.eva_history()
    resp = jsonify(errno=task.error_code, data=task.process)
    return resp


@api.route('/starteva', methods=['POST'])
def starteva():
    """
    l_value: 是否评估测试数据；
    type: 挑选数据，人工数据的文件夹；
    :return:
    """

    req_dict = request.json
    dicts = req_dict.get('dicts')
    l_value = req_dict.get('l_value')
    type = req_dict.get('type')
    gpus = req_dict.get('gpus')
    host = req_dict.get('host')
    single_gpu = req_dict.get('single_gpu')
    img = req_dict.get('img')
    output_confidence = req_dict.get("output_confidence")
    task_id = uuid.uuid1()
    status = 'starting'
    if host != mach_id:
        url = url_mach[host]
        url = url + "starteva"
        data = {
            'dicts': dicts,
            "l_value": l_value,
            "type": type,
            "gpus": gpus,
            "host": host,
            "single_gpu": single_gpu,
            "img": img,
            "output_confidence": output_confidence
        }
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        if js_data['errno'] == 1:
            resp = jsonify(errno=js_data['errno'],
                           message=js_data['message'],
                           data={"task_id": js_data['data']["task_id"]})
        else:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
    else:
        sour_li, dest, models, task_type = copy_data(l_value, dicts, mach_id, type)

        if all([sour_li, dest, models, task_type]):
            eva = EvaTasks()
            eva.evaluating(task_id, sour_li, gpus, dest, single_gpu, models, status, task_type, img, output_confidence, host)
            resp = jsonify(errno=eva.error_code, message=eva.message, data={"task_id": task_id})
        else:
            resp = jsonify(errno=0, message='no found model')
    return resp


@api.route('/show', methods=["POST"])
def show():
    """
    :return:
    """
    req_dict = request.json
    task_id = req_dict.get('task_id')
    cur_img = req_dict.get('cur_img')
    cur_img = int(cur_img)
    lists = []
    or_data = None
    eva_task = BaseEvaTask()
    try:
        sour_dir, dest_dir, host = eva_task.eva_query(task_id)
        if host != mach_id:
            url = url_mach[host]
            url = url + "show"
            data = {
                'task_id': task_id,
                "cur_img": cur_img,
            }
            json_datas = requests.post(url=url, json=data)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:

                resp = jsonify(errno=js_data['errno'],
                               or_data=js_data['or_data'],
                               la_data=js_data['la_data'],
                               total_img=js_data['total_img'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:

            data_json_str = redis_store.get(task_id)
            if data_json_str:
                lists.extend(json.loads(data_json_str))
            else:
                sour_dir = sour_dir.split(',')
                sour_dir = list(set(sour_dir[:-1]))
                # model = list(model)
                if sour_dir and dest_dir:
                    for sour in sour_dir:
                        if sour:
                            img_list = os.listdir(sour)
                            for img_name in img_list:
                                if img_name.endswith("jpg"):
                                    dicts = {}
                                    img_path = os.path.join(sour, img_name)
                                    lis = os.listdir(dest_dir)
                                    img = img_name[:-4]
                                    label_name = img + '.png'
                                    label_path_li = []
                                    for li in lis:
                                        label_path_li.append(os.path.join(dest_dir, li, label_name))

                                    dicts[img_path] = label_path_li
                                    lists.append(dicts)

                    try:
                        redis_store.set(task_id, json.dumps(lists), 10800)
                    except Exception as e:
                        current_app.logger.error(e)
            total_img = len(lists)
            img_data = lists[cur_img - 1]
            la_datas = []
            for k, v in img_data.items():
                with open(k, 'rb') as f1:
                    or_data = base64.b64encode(f1.read())
                for i in v:
                    with open(i, 'rb') as f2:
                        la_data = base64.b64encode(f2.read())
                        ss = i.split('/')
                        s_dicts = {}
                        s_dicts['name'] = ss[-2]
                        s_dicts['data'] = la_data
                        la_datas.append(s_dicts)
            resp = jsonify(errno=1, or_data=or_data, la_data=la_datas, total_img=total_img)
    except Exception as e:
        resp = jsonify(errno=0, message='no task id')
        current_app.logger.error(e)
    finally:
        return resp


@api.route('/show_info', methods=["POST"])
def show_info():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    eva_task = BaseEvaTask()
    info_data = ""
    try:
        sour_dir, dest_dir, host = eva_task.eva_query(task_id)
        if host != mach_id:
            url = url_mach[host]
            url = url + "show_info"
            data = {'task_id': task_id}
            json_datas = requests.post(url=url, json=data)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:

                resp = jsonify(errno=js_data['errno'], data=js_data['data'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])

        else:
            lis = os.listdir(dest_dir)
            for i in lis:
                l_path = os.path.join(dest_dir, i, "info.txt")
                t_path = os.path.join(dest_dir, i, "trt_info.txt")
                if os.path.exists(l_path):
                    with open(l_path, 'r') as f:
                        info_data += i + "\n" + f.read() + "\n\n\n"
                if os.path.exists(t_path):
                    with open(t_path, 'r') as f:
                        info_data += i + "\n" + f.read() + "\n\n\n"
            resp = jsonify(errno=1, data=info_data)
    except Exception as e:
        resp = jsonify(errno=0, message='no task id')
        current_app.logger.error(e)
    return resp


@api.route('/eva_data', methods=["GET"])
def eva_data():
    """
    :return:
    """
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    lists = []
    path = '/data/deeplearning/train_platform/eva_sour_data/s_data'
    name_li = dest_sftp.listdir(path)
    for i in name_li:
        if i != 'test_data':
            lists.append(i)
    lists.insert(0, "请选择数据")
    resp = jsonify(errno=1, data=lists)
    return resp


@api.route('/con_info', methods=["POST"])
def con_info():
    req_dict = request.json
    task_name = req_dict.get('task_name')
    model = req_dict.get('model')
    con_data = db.session.query(Confidence_Datas).filter_by(task_name=task_name, model=model).all()
    con_list = []
    total_o_con = 0
    total_con = 0
    total_cls_con = {}

    if con_data:
        len_con = len(con_data)
        for con in con_data:
            con_list.append(con.to_dict())
        for li in con_list:
            total_o_con += float(li["origin_whole_con"])
            total_con += float(li["whole_con"])
            for k, v in json.loads(str(li["origin_cls_con"]).replace("'", "\"")).items():
                if k not in total_cls_con:
                    if v == 'nan':
                        v = 0
                    total_cls_con[k] = float(v)
                else:
                    if v == 'nan':
                        v = 0
                    total_cls_con[k] += float(v)
        total_o_con = round(total_o_con / len_con, 4)
        total_con = round(total_con / len_con, 4)
        for k, v in total_cls_con.items():
            total_cls_con[k] = round(v / len_con, 4)

        resp = jsonify(errno=1,
                       con_data=con_list,
                       total_data={
                           "整体置信度": total_o_con,
                           "整体置信度（去掉路面和其他）": total_con,
                           "total_cls_con": total_cls_con
                       })
    else:
        resp = jsonify(errno=0, message="No data")
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
