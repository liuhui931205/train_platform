# -*-coding:utf-8-*-
from Apps.models import CheckDatas
from Apps import db
from flask import current_app
import re


class BaseCkeckTask(object):

    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, task_name, gpus, task_id, weights_dir, status):
        checkdata = CheckDatas()
        checkdata.task_id = task_id
        checkdata.task_name = task_name
        checkdata.gpus = gpus
        checkdata.weights_dir = weights_dir
        checkdata.status = status
        try:
            db.session.add(checkdata)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def task_history(self):
        task_hitys = db.session.query(CheckDatas).all()
        self.process = []
        for task_hity in task_hitys:
            self.process.append(task_hity.to_dict())
        return self.process

    def task_status(self, task_id):
        auto_select = db.session.query(CheckDatas).filter_by(task_id=task_id).first()
        self.process = []
        self.error_code = 1
        self.process.append(auto_select.to_dict())

    def callback(self, task_id, status):
        check_data = db.session.query(CheckDatas).filter_by(task_id=task_id).first()
        vs = check_data.status
        value = 0
        if vs == 'starting':
            check_data = db.session.query(CheckDatas).filter_by(task_id=task_id).first()
            check_data.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    check_data = db.session.query(CheckDatas).filter_by(task_id=task_id).first()
                    check_data.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        check_data = db.session.query(CheckDatas).filter_by(task_id=task_id).first()
                        check_data.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(check_data)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
