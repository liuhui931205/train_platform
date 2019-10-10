# -*- coding:utf-8 -*-
from Apps.models import TaskDivide
from Apps import db
from flask import current_app
import re


class BaseDivideTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def starts(self, version, step, types, task_id, status):
        task_divide = TaskDivide()
        task_divide.task_id = task_id
        task_divide.version = version
        task_divide.step = step
        task_divide.types = types
        task_divide.status = status

        try:
            db.session.add(task_divide)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def divide_history(self):
        divide_hitys = db.session.query(TaskDivide).all()
        self.process = []
        self.error_code = 1
        for divide_hity in divide_hitys:
            self.process.append(divide_hity.to_dict())

    def divide_status(self, task_id):
        divide_task = db.session.query(TaskDivide).filter_by(task_id=task_id).first()
        if divide_task:
            self.process = []
            self.error_code = 1
            self.process.append(divide_task.to_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        divide_task = db.session.query(TaskDivide).filter_by(task_id=task_id).first()
        vs = divide_task.status
        value = 0
        if vs == 'starting':
            divide_task = db.session.query(TaskDivide).filter_by(task_id=task_id).first()
            divide_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    divide_task = db.session.query(TaskDivide).filter_by(task_id=task_id).first()
                    divide_task.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        divide_task = db.session.query(TaskDivide).filter_by(task_id=task_id).first()
                        divide_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(divide_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
