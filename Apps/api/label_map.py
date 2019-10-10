# -*- coding:utf-8 -*-
from . import api
from flask import jsonify, request
from Apps.modules.base_train import BaseTrainTask
from config import url_mach,data_sour_train
from Apps import mach_id
import requests
import sys
import copy
import json
import os
from Apps.utils.utils import seg_label, arrow_labels, self_full_labels, object_labels_sign

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/save_map', methods=['POST'])
def save_map():
    req_dict = request.json
    map_value = req_dict.get('map_value')
    task_id = req_dict.get('task_id')
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    if host != mach_id:
        url = url_mach[host]
        url = url + "save_map"
        data = {'map_value': map_value, "task_id": task_id}
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        if js_data['errno'] == 1:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
        else:
            resp = jsonify(errno=js_data['errno'], message=js_data['message'])
    else:
        if task_type == "target-detection":
            lits = []
            for label in object_labels_sign:
                di = {}
                di["color"] = label.color
                di["className"] = label.en_name
                di["trainId"] = label.trainId
                di["categoryId"] = label.categoryId
                lits.append(di)
            map_paths = os.path.join(data_sour_train, task_name, "output/models")
            json.dump(lits, open(os.path.join(map_paths, 'map_label.json'), 'w'), encoding='utf-8')
            resp = jsonify(errno=1, message='save success')
        else:
            map_v = copy.deepcopy(map_value)
            dicts = {}
            lis2 = []
            lis = []
            li = []
            for k, v in map_value.items():
                if k not in lis:
                    if int(v[3]) != 255:
                        j = v[3]
                        if li:
                            s = li[-1] + 1
                        else:
                            s = 0
                        li.append(s)
                        s = str(s)
                        for q, w in map_value.items():
                            if map_value[q][3] == j and q not in lis:
                                map_value[q][3] = s
                                lis.append(q)
            for k, v in map_v.items():
                lid = []
                if k == '20':
                    lid.append(u'虚拟车道线-路缘石')
                    lid.append((0, 139, 139))
                    dicts['254'] = lid
                else:
                    if v[3] not in lis2:
                        lis2.append(v[3])

                        for label in seg_label:
                            if int(label.categoryId) == int(v[3]):
                                lid.append(label.name)
                                lid.append(label.color)
                        dicts[map_value[k][3]] = lid
            map_paths = os.path.join(data_sour_train, task_name, "output/models")
            # json.dump(dicts, open(os.path.join(map_paths, 'label.json'), 'w'), encoding='utf-8')
            lits = []
            for k, v in dicts.items():
                di = {}
                for label in self_full_labels:
                    if v[0] == label.name or tuple(v[1]) == label.color:
                        di["color"] = v[1]
                        di["en_name"] = label.en_name
                        di["id"] = label.id
                        di["name"] = v[0]
                        di["categoryId"] = k
                        lits.append(di)
                        break
            json.dump(lits, open(os.path.join(map_paths, 'map_label.json'), 'w'), encoding='utf-8')

            data = '#id\tcategoryId\ten_name\tname\n'
            for k, v in map_value.items():
                data += k + '\t'
                data += v[3] + '\t'
                data += v[1] + '\t'
                data += v[2] + '\n'
            map_path = os.path.join(data_sour_train, task_name, 'conf/label_map.txt')
            with open(map_path, 'w') as f:
                f.write(data)
            resp = jsonify(errno=1, message='save success')
    return resp


@api.route('/maps', methods=['POST'])
def label_map():
    li = []
    id_list = []
    req_dict = request.json
    task_id = req_dict.get('task_id')
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    if host != mach_id:
        url = url_mach[host]
        url = url + "maps"
        data = {"task_id": task_id}
        json_datas = requests.post(url=url, json=data)
        js_data = json.loads(json_datas.text)
        resp = jsonify(errno=js_data['errno'], data=js_data['data'])
    else:
        map_path = os.path.join(data_sour_train, task_name, 'conf/label_map.txt')
        with open(map_path, 'r') as f:
            while True:
                line = f.readline()
                if line:
                    line = line.split('\t')
                    li.append(line)
                else:
                    break
        for i in li:
            if li.index(i) != 0:
                value = i[0]
                if value != '\n':
                    id_dict = {}
                    lis = []
                    lis.append(i[1].strip('\n'))
                    lis.append(i[2].strip('\n'))
                    lis.append(i[3].strip('\n'))
                    id_dict[int(value)] = lis
                    id_list.append(id_dict)
        id_list.sort()
        resp = jsonify(errno=1, data=id_list)
    return resp


@api.route('/seg_maps', methods=['GET'])
def seg_maps():
    task_id = request.args.get('task_id')
    task = BaseTrainTask()
    task_name, task_type, host = task.task_status(task_id)
    li = []
    if host != mach_id:
        url = url_mach[host]
        url = url + "seg_maps?task_id={}".format(task_id)
        json_datas = requests.get(url=url)
        js_data = json.loads(json_datas.text)
        resp = jsonify(errno=js_data['errno'], data=js_data['data'])

    else:
        if task_type == "semantic-segmentation":
            for label in seg_label:
                dicts = {}
                dicts["name"] = label.name
                dicts["value"] = label.categoryId
                li.append(dicts)
        elif task_type == "classification-model":
            for label in arrow_labels:
                dicts = {}
                dicts["name"] = label.name
                dicts["value"] = label.categoryId
                li.append(dicts)
        elif task_type == "target-detection":
            for label in object_labels_sign:
                dicts = {}
                dicts["name"] = label.className
                dicts["value"] = label.categoryId
                li.append(dicts)
        resp = jsonify(errno=1, data=li)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
