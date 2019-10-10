# -*- coding:utf-8 -*-
from . import api
import time
from flask import jsonify, request, current_app
import os
from Apps.utils.client import client
import json
import requests
from Apps.models import Released_Models
from stat import S_ISDIR
from Apps import db, mach_id
import sys
from Apps.models import TrainTask
from config import data_sour_train, data0_sour_train
import uuid

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/release', methods=["POST"])
def release_new():
    model_dict = None
    model_json = []
    req_dict = request.json
    version = req_dict.get('version')
    env = req_dict.get('env')
    adcode = req_dict.get('model_code')
    desc = req_dict.get('desc')
    times = req_dict.get('time')
    types = req_dict.get('type')
    dicts = req_dict.get('dicts')
    page = req_dict.get('current_page')
    prod = req_dict.get("prod")
    if prod:
        prods = "1"
    else:
        prods = "0"
    if not page:
        for k, v in dicts.items():
            if v:
                model_path = '/data/deeplearning/train_platform/train_task/{}/output/models'.format(k)
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                train_task = db.session.query(TrainTask).filter_by(task_name=k).first()
                if train_task:
                    status = train_task.status
                    if status == "completed!":
                        dest_scp, dest_sftp, dest_ssh = client(id="host")
                        model_spath = os.path.join(data0_sour_train, k, "output/models")
                        for i in v:
                            model_dict = os.path.join(model_path, i)
                            if not os.path.exists(model_dict):
                                dest_sftp.get(os.path.join(model_path, i), model_dict)
                        li = dest_sftp.listdir(model_spath)
                        for i in li:
                            if i == "map_label.json" or i.endswith("symbol.json") or i.endswith('.plan'):
                                json_path = os.path.join(model_path, i)
                                if not os.path.exists(json_path):
                                    dest_sftp.get(os.path.join(model_path, i), json_path)
                                model_json.append(json_path)
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()

    else:
        page = int(page)
        if page == 1:
            types = 'lane-resnet'

        elif page == 2:
            types = 'sign-mask-rcnn'

        elif page == 3:
            types = 'pole-mxnet'

        elif page == 4:
            types = 'virtual-lane'

        elif page == 5:
            types = 'arrow-classfication'

        elif page == 6:
            types = 'flownet'

        elif page == 7:
            types = 'sign-pspnet'

        for k, v in dicts.items():
            if k != 'current_page':
                for i in v:
                    if i.endswith('.params'):
                        model_dict = '/data/model/' + types + '/' + k + '/' + i
                    elif i == "map_label.json" or i.endswith("symbol.json"):
                        model_json.append('/data/model/' + types + '/' + k + '/' + i)
    task_id = uuid.uuid1()
    if not times:
        times = time.strftime("%Y/%m/%d %H:%M", time.localtime())
    release_model = Released_Models()
    release_model.time = times
    release_model.task_id = task_id
    release_model.type = types
    release_model.adcode = adcode
    release_model.desc = desc
    release_model.env = env
    release_model.version = version
    for k, v in dicts.items():
        release_model.tasks_name = k
        for i in v:
            if i.endswith('.params'):
                release_model.model_name = i
    try:
        db.session.add(release_model)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(e)
        resp = jsonify(errno='0', message='save failed')
        return resp
    try:
        dest_scp, dest_sftp, dest_ssh = client(id='host')
    except Exception as e:
        current_app.logger.error(e)

    if types == 'resnet-road':
        modelPath = os.path.join('/data/model/lane-resnet', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/lane-resnet', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/lane-resnet', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/lane-resnet', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/lane-resnet', version, os.path.split(i)[1]))
    elif types == 'sign-mask-rcnn':
        modelPath = os.path.join('/data/model/sign-mask-rcnn', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/sign-mask-rcnn', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/sign-mask-rcnn', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/sign-mask-rcnn', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/sign-mask-rcnn', version, os.path.split(i)[1]))
    elif types == 'resnet1':
        modelPath = os.path.join('/data/model/pole-mxnet', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/pole-mxnet', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/pole-mxnet', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/pole-mxnet', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/pole-mxnet', version, os.path.split(i)[1]))
    elif types == 'virtual-lane':
        modelPath = os.path.join('/data/model/virtual-lane', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/virtual-lane', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/virtual-lane', version))

        dest_sftp.put(model_dict, os.path.join('/data/model/virtual-lane', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/virtual-lane', version, os.path.split(i)[1]))
    elif types == 'arrow':
        modelPath = os.path.join('/data/model/arrow-classfication', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/arrow-classfication', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/arrow-classfication', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/arrow-classfication', version,
                                               os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/arrow-classfication', version, os.path.split(i)[1]))
    elif types == 'flow-net':
        modelPath = os.path.join('/data/model/flownet', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/flownet', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/flownet', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/flownet', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/flownet', version, os.path.split(i)[1]))
    elif types == 'pspnet':
        modelPath = os.path.join('/data/model/sign-pspnet', version, os.path.split(model_dict)[1])
        try:
            dest_sftp.stat(os.path.join('/data/model/sign-pspnet', version))
        except IOError:
            dest_sftp.mkdir(os.path.join('/data/model/sign-pspnet', version))
        dest_sftp.put(model_dict, os.path.join('/data/model/sign-pspnet', version, os.path.split(model_dict)[1]))
        for i in model_json:
            dest_sftp.put(i, os.path.join('/data/model/sign-pspnet', version, os.path.split(i)[1]))

    # for i in model_json:
    #     if i.endswith('.plan'):
    #         model_name = os.path.basename(i)
    #         modelPath = modelPath + ','+os.path.join(os.path.dirname(modelPath),model_name)

    info_csv_src = "/data/model/info.csv"
    info_csv_dest = "/data/deeplearning/train_platform/info.csv"
    dest_sftp.get(info_csv_src, info_csv_dest)
    last_str = ""
    str_obj = []
    with open(info_csv_dest, "r") as f:
        line_str = f.readline()
        while line_str:
            if line_str:
                last_str = line_str
                str_obj.append(line_str)
                line_str = f.readline()
    with open(info_csv_dest, "w") as f:
        if str_obj:
            for line_str in str_obj:
                f.write(line_str)
        else:
            last_str = '\n'
        if prod == "1":
            if last_str[-1:] != '\n':
                input_str = "\n{},{},{},{},{},{},{}\n".format(modelPath, types, env, version, str(times), desc, adcode)
                input_str += "{},{},prod,{},{},{},{}\n".format(modelPath, types, version, str(times), desc, adcode)
            else:
                input_str = "{},{},{},{},{},{},{}\n".format(modelPath, types, env, version, str(times), desc, adcode)
                input_str += "{},{},prod,{},{},{},{}\n".format(modelPath, types, version, str(times), desc, adcode)
        else:
            if last_str[-1:] != '\n':
                input_str = "\n{},{},{},{},{},{},{}\n".format(modelPath, types, env, version, str(times), desc, adcode)
            else:
                input_str = "{},{},{},{},{},{},{}\n".format(modelPath, types, env, version, str(times), desc, adcode)

        f.write(input_str)

    dest_sftp.put(info_csv_dest, info_csv_src)

    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    resp = jsonify(errno=1, message=u'发布成功', data={"task_id": task_id})

    return resp


@api.route('/release_tab', methods=['GET'])
def release_tab():
    datas = {}
    page = request.args.get('page')
    page = int(page)

    if page == 1:
        types = 'lane-resnet'
        name = u'车道线'
    elif page == 2:
        types = 'sign-mask-rcnn'
        name = u'路牌'
    elif page == 3:
        types = 'pole-mxnet'
        name = u'灯杆'
    elif page == 4:
        types = 'virtual-lane'
        name = u'虚拟车道线'
    elif page == 5:
        types = 'arrow-classfication'
        name = u'箭头识别'
    elif page == 6:
        types = 'flownet'
        name = u'光流'
    elif page == 7:
        types = 'sign-pspnet'
        name = u'旧路牌'
    try:
        dest_scp, dest_sftp, dest_ssh = client(id='host')
        path = os.path.join('/data/model', types)
        li = dest_sftp.listdir(path)
    except Exception as e:
        current_app.logger.error(e)
    type_list = []

    for i in li:
        if S_ISDIR(dest_sftp.stat(os.path.join('/data/model/', types, i)).st_mode):
            type_list.append(i)
    total = 7
    for li in type_list:
        model_list = []
        mod_li = dest_sftp.listdir(os.path.join('/data/model/', types, li))
        for j in mod_li:
            if j.endswith('.params'):
                model_list.append(j)
        datas[li] = model_list
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    resp = jsonify(errno=1, data=datas, name=name, total=total)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
