# -*- coding:utf-8 -*-
from . import api
import os
from Apps.modules.base_train import BaseTrainTask
from flask import jsonify, request, current_app
import sys
from Apps.utils.client import client
from Apps import mach_id
from config import data_sour_train, data0_sour_train

reload(sys)
sys.setdefaultencoding('utf-8')


@api.route('/model', methods=['GET'])
def model_history():
    page = request.args.get('page')
    page = int(page)
    total = 4
    if page == 1:
        types = 'semantic-segmentation'
        name = u'语义分割'
    elif page == 2:
        types = 'classification-model'
        name = u'分类模型'
    elif page == 3:
        types = 'target-detection'
        name = u'目标检测'
    elif page == 4:
        types = 'OCR'
        name = u'OCR'
    datas = {}

    tasks = BaseTrainTask()
    tasks.task_history()
    # tasks.process.sort(reverse=True)
    for task in tasks.process:
        models_li = []
        machineid = task['machine_id']
        task_type = task['type']
        status = task["status"]
        if task_type == types:
            task_name = task['task_name']
            if status != "completed!":
                mod_path = os.path.join(data_sour_train, task_name, "output/models")
                if machineid != mach_id:
                    try:
                        dest_scp, dest_sftp, dest_ssh = client(id=machineid)
                        mod_li = dest_sftp.listdir(mod_path)
                        for i in mod_li:
                            if i.endswith('.params'):
                                models_li.append(i)
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()
                    except Exception as e:
                        current_app.logger.error(e)
                else:
                    if os.path.exists(mod_path):
                        mod_li = os.listdir(mod_path)
                        for i in mod_li:
                            if i.endswith('.params'):
                                models_li.append(i)
            else:
                try:
                    mod_path = os.path.join(data0_sour_train, task_name, "output/models")
                    dest_scp, dest_sftp, dest_ssh = client(id="host")

                    mod_li = dest_sftp.listdir(mod_path)
                    for i in mod_li:
                        if i.endswith('.params'):
                            models_li.append(i)
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                except Exception as e:
                    current_app.logger.error("{},{}".format(task_name,e))

            if len(models_li) > 0:
                models_li.sort(reverse=True)
            datas[task_name] = models_li
    resp = jsonify(errno=1, data=datas, total=total, name=name)

    return resp


@api.route('/t_model', methods=['GET'])
def t_model():
    page = request.args.get('page')
    page = int(page)
    total = 4
    if page == 1:
        types = 'semantic-segmentation'
        name = u'语义分割'
    elif page == 2:
        types = 'classification-model'
        name = u'分类模型'
    elif page == 3:
        types = 'target-detection'
        name = u'目标检测'
    elif page == 4:
        types = 'OCR'
        name = u'OCR'
    datas = {}
    tasks = BaseTrainTask()
    tasks.task_history()
    # tasks.process.sort(reverse=True)
    for task in tasks.process:
        models_li = []
        machineid = task['machine_id']
        task_type = task['type']
        status = task["status"]
        if task_type == types:
            task_name = task['task_name']
            if status != "completed!":
                mod_path = os.path.join(data_sour_train, task_name, "output/models")
                if machineid != mach_id:
                    try:
                        dest_scp, dest_sftp, dest_ssh = client(id=machineid)
                        mod_li = dest_sftp.listdir(mod_path)
                        for i in mod_li:
                            if i.endswith('.params') or i.endswith('.plan'):
                                models_li.append(i)
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()
                    except Exception as e:
                        current_app.logger.error(e)
                else:
                    if os.path.exists(mod_path):
                        mod_li = os.listdir(mod_path)
                        for i in mod_li:
                            if i.endswith('.params') or i.endswith('.plan'):
                                models_li.append(i)
            else:
                mod_path = os.path.join(data0_sour_train, task_name, "output/models")
                dest_scp, dest_sftp, dest_ssh = client(id="host")
                mod_li = dest_sftp.listdir(mod_path)
                for i in mod_li:
                    if i.endswith('.params') or i.endswith('.plan'):
                        models_li.append(i)
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
            if len(models_li) > 0:
                models_li.sort(reverse=True)
            datas[task_name] = models_li
    resp = jsonify(errno=1, data=datas, total=total, name=name)
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
