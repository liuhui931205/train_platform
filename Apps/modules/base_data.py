# -*- coding:utf-8 -*-
from Apps.models import Datas
from Apps import db, mach_id
from flask import current_app
import re


class BaseDataTask(object):

    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, taskid, task_id, data_name, data_type, train, val, test, sour_data, data_describe, status, types,
              image_type):
        data = Datas()
        data.data_name = data_name
        data.task_id = task_id
        data.data_type = data_type
        data.train = train
        data.test = test
        data.val = val
        if taskid:
            data_task = db.session.query(Datas).filter_by(task_id=taskid).first()
            sour_data = data_task.to_data_dict()["sour_data"] + "," + sour_data
        data.sour_data = sour_data
        data.data_describe = data_describe
        data.status = status
        data.machine_id = mach_id
        data.type = types
        data.images_type = image_type
        try:
            db.session.add(data)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def data_history(self):
        data_hitys = db.session.query(Datas).order_by(Datas.id.desc()).all()
        self.process = []
        for data_hity in data_hitys:
            self.process.append(data_hity.to_data_dict())

    def data_status(self, task_id):
        data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
        if data_task:
            self.process = []
            self.error_code = 1
            self.process.append(data_task.to_data_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
        vs = data_task.status
        value = 0
        if vs == 'starting':
            data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
            data_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
                    data_task.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        data_task = db.session.query(Datas).filter_by(task_id=task_id).first()
                        data_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(data_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
