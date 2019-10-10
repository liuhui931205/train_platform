# -*-coding:utf-8-*-
from . import api
from flask import jsonify, request, current_app
from Apps.modules.score_task import ScoreTasks
from Apps.modules.base_score import BaseScoreTask
from Apps import mach_id
import uuid
import sys
import os
import base64
import requests
import json
from config import url_mach

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/record_score', methods=["POST"])
def record_score():
    req_dict = request.json
    task_id = req_dict.get('task_id')
    score = req_dict.get('score')
    standard = req_dict.get('standard')
    task_name = req_dict.get('task_name')
    task = BaseScoreTask()
    task.data_update(task_id, {"status": "completed!", "score": score, "standard": standard, "task_name": task_name})
    resp = jsonify(errno=1, message='success')
    return resp


@api.route('/show_recog', methods=["POST"])
def show_recog():
    """
    :return:
    """
    req_dict = request.json
    task_id = req_dict.get('task_id')
    cur_img = req_dict.get('cur_img')
    cur_img = int(cur_img)
    score_task = ScoreTasks()
    try:
        result = score_task.data_status(task_id)
        host = result["host"]
        if host != mach_id:
            url = url_mach[host]
            url = url + "show_recog"
            data = {'task_id': task_id, "cur_img": cur_img}
            json_datas = requests.post(url=url, json=data)
            js_data = json.loads(json_datas.text)
            if js_data['errno'] == 1:
                resp = jsonify(errno=js_data['errno'],
                               or_data=js_data['or_data'],
                               la_datas=js_data['la_datas'],
                               total_img=js_data['total_img'])
            else:
                resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            lists = []
            area_name = result["area_name"]
            scence_name = result["scence_name"]
            png_path = result["png_path"].split(",")[:-1]
            sour_dir = os.path.join("/data/deeplearning/train_platform/recog_rate_sour", area_name, scence_name)
            dest_dir = os.path.join("/data/deeplearning/train_platform/recog_rate_dest", area_name, scence_name)
            files_list = os.listdir(sour_dir)
            for img_name in files_list:
                if img_name.endswith("jpg"):
                    dicts = {}
                    img_path = os.path.join(sour_dir, img_name)
                    label_name = img_name[:-3] + 'png'
                    label_path_li = []
                    for li in png_path:
                        label_path_li.append(os.path.join(dest_dir, li, label_name))
                    dicts[img_path] = label_path_li
                    lists.append(dicts)

            lists.sort()
            total_img = len(lists)
            la_datas = []
            name = files_list[cur_img - 1]
            for k, v in name.items():
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
    return resp


# 识别打分
@api.route('/recog_rate', methods=["POST"])
def recog_rate():
    req_dict = request.json
    area_name = req_dict.get('area_name')
    weights_dir = req_dict.get('weights_dir')
    gpus = req_dict.get('gpus')
    img = req_dict.get('img_type')
    scence_name = req_dict.get('scence_name')
    host = req_dict.get('host')
    if host != mach_id:
        url = url_mach[host]
        url = url + "recog_rate"
        data = {
            'area_name': area_name,
            "weights_dir": weights_dir,
            "gpus": gpus,
            "img_type": img,
            "host": host,
            "scence_name": scence_name
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
        task_id = uuid.uuid1()
        status = "starting"
        score_task = ScoreTasks()
        score_task.do_task(task_id, area_name, scence_name, gpus, weights_dir, img, status, host)
        resp = jsonify(errno=score_task.error_code, message=score_task.message, data={"task_id": task_id})
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
