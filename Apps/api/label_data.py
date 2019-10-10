# -*-coding:utf-8-*-
from . import api
from Apps.models import LabelData, Schedulefrom
from Apps import db, redis_store
import json
from Apps.tasks.tasks import store_data
from flask import jsonify, request, current_app
from Apps.utils.utils import self_full_labels
from Apps.libs.query_datas import filter_data
import time
import uuid


# 创建数据导入任务
@api.route("/label_data", methods=["POST"])
def label_data():
    req_dict = request.json
    picid = req_dict.get('picid')
    id_code = uuid.uuid1()
    task = store_data.apply_async(args=[picid, id_code], countdown=5)    # 发送异步任务，指定队列
    try:
        data = Schedulefrom()
        data.task_id = task.id
        data.id_code = id_code
        data.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        data.picid = picid
        data.statue = "pending"
        db.session.add(data)
        db.session.commit()
    except Exception as e:
        current_app.logger.error(e)
    resp = jsonify(errno=1, message="success")
    return resp


# 分页标注数据查询
@api.route("/query_data", methods=["POST"])
def query_data():
    req_dict = request.json
    imgrange = str(req_dict.get('imgrange'))
    city = str(req_dict.get('city'))
    label_info = str(req_dict.get('label_info'))
    tag_info = req_dict.get('tag_info')
    pacid = req_dict.get('pacid')
    pag_id = req_dict.get('pag_id')
    if not pag_id:
        pag_id = 1
    pag_id = int(pag_id)
    try:
        data_json_str = redis_store.get('label_data')
        if data_json_str:
            data_dict_li = json.loads(data_json_str)
            querydata = filter_data(
                data_dict_li, imgrange=imgrange, city=city, label_info=label_info, tag_info=tag_info, pacid=pacid)
            count = len(querydata)

            querydata = querydata[(pag_id - 1) * 15:pag_id * 15]
            return jsonify(errno=1, message='success', data=querydata, count=count)
    except Exception as e:
        current_app.logger.error(e)
    try:
        con_data = db.session.query(LabelData).all()
    except Exception as e:
        current_app.logger.error(e)
        return jsonify(errno=1, message='query failed')

    data_dict_li = []
    for data in con_data:
        data_dict_li.append(data.to_dict())
    querydata = filter_data(
        data_dict_li, imgrange=imgrange, city=city, label_info=label_info, tag_info=tag_info, pacid=pacid)
    count = len(querydata)

    querydata = querydata[(pag_id - 1) * 15:pag_id * 15]

    try:
        redis_store.set('label_data', json.dumps(data_dict_li), 10800)
    except Exception as e:
        current_app.logger.error(e)
    return jsonify(errno=1, message='success', data=querydata, count=count)


# 标签
@api.route("/label_infos", methods=["GET"])
def label_infos():
    label_map = {}
    for label in self_full_labels:
        label_map[label.name] = label.id
    resp = jsonify(errno=1, data=label_map)
    return resp


# 数据导入进度查询
@api.route('/label_status', methods=["GET"])
def taskstatus():
    task_id = request.args.get('task_id')
    task = store_data.AsyncResult(task_id)
    if task:
        if task.state == 'PROGRESS':
            data = {
                'state': task.state,
                'success': task.info.get("success"),
                'failed': task.info.get("failed"),
                'total': task.info.get("total"),
                'status': task.info.get("status")
            }
            resp = jsonify(errno=1, message="progress", data=data)
        else:
            data_task = db.session.query(Schedulefrom).filter_by(task_id=task_id).first()
            if data_task.statue == "failure" or data_task.statue.startswith("completed"):
                resp = jsonify(errno=1, message=data_task.statue)
            else:
                resp = jsonify(errno=1, message="please try later")
    else:
        data_task = db.session.query(Schedulefrom).filter_by(task_id=task_id).first()
        if data_task.statue == "failure" or data_task.statue.startswith("completed"):
            resp = jsonify(errno=1, message=data_task.statue)

        else:
            resp = jsonify(errno=1, message="please try later")
    return resp


# 数据导入历史任务
@api.route('/label_tasks', methods=["GET"])
def label_tasks():
    tasks_list = []
    data_hitys = db.session.query(Schedulefrom).order_by(Schedulefrom.id.desc()).all()
    for data_hity in data_hitys:
        tasks_list.append(data_hity.to_dict())
    resp = jsonify(errno=1, data=tasks_list)
    return resp


# 色板信息
@api.route('/class_info', methods=["GET"])
def class_info():
    info = ["1.0", "4.13", "5.1", "6.1"]
    resp = jsonify(errno=1, data=info)
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
