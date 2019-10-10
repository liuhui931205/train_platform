# -*- coding:utf-8 -*-
from . import api
import linecache
from Apps.utils.read_log import get_line_count
from flask import jsonify, current_app, request
from Apps.models import TrainTask, Datas
from Apps import db, mach_id
import sys
from config import data_sour_train, data0_sour_train
from Apps.utils.client import client
import os

reload(sys)
sys.setdefaultencoding('utf-8')

seek = int(0)


@api.route('/output_log', methods=["GET"])
def connected_msg():
    # 日志存放位置
    task_id = request.args.get('task_id')
    train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
    task_name = train_task.task_name
    model = train_task.model
    mac_id = train_task.machine_id
    status = train_task.status

    data_id = train_task.data_desc_id
    data_task = db.session.query(Datas).filter_by(id=data_id).first()
    data_name = data_task.data_name
    if mac_id != mach_id:
        r_log = os.path.join(data_sour_train, task_name, "log", "{}.train.log.{}".format(model, data_name))
        if status != "completed!":

            dest_scp, dest_sftp, dest_ssh = client(id=mac_id)
            src_log = os.path.join(data_sour_train, task_name, "log", "{}.train.log.{}".format(model, data_name))

        else:
            dest_scp, dest_sftp, dest_ssh = client(id="host")
            src_log = os.path.join(data0_sour_train, task_name, "log", "{}.train.log.{}".format(model, data_name))
        path = os.path.join(data_sour_train, task_name, "log")
        if not os.path.exists(path):
            os.makedirs(path)

        dest_sftp.get(src_log, r_log)
        dest_sftp.close()
        dest_scp.close()
        dest_ssh.close()

    else:
        r_log = os.path.join(data_sour_train, task_name, "log", "{}.train.log.{}".format(model, data_name))
    if os.path.exists(r_log):
        try:
            n = 100
            data = []
            linecache.clearcache()
            line_count = get_line_count(r_log)
            line_count -= n
            for i in range(n + 1):    # the last 30 lines
                last_line = linecache.getline(r_log, line_count)
                data.append(last_line)
                line_count += 1
            resp = jsonify(errno=1, data=data)
        except Exception as e:
            current_app.logger.error(e)
            resp = jsonify(errno=0, message='failed')
    else:
        resp = jsonify(errno=0, message='wait a moment')
    return resp


@api.after_request
def af_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'OPTIONS,GET,POST'
    response.headers['Access-Control-Max-Age'] = '1800'
    response.headers['Access-Control-Allow-Headers'] = 'x-requested-with,content-type'
    return response
