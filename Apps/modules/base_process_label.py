# -*- coding:utf-8 -*-
from Apps.models import LabelProcess
from Apps import db
from flask import current_app
import re


class BaseProcessLabel(object):

    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def starts(self, version, name, types, task_id, status, color_info):
        process_label = LabelProcess()
        process_label.task_id = task_id
        process_label.version = version
        process_label.name = name
        process_label.types = types
        process_label.status = status
        process_label.color_info = color_info

        try:
            db.session.add(process_label)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def label_history(self):
        label_hitys = db.session.query(LabelProcess).all()
        self.process = []
        self.error_code = 1
        for label_hity in label_hitys:
            self.process.append(label_hity.to_dict())

    def label_status(self, task_id):
        label_task = db.session.query(LabelProcess).filter_by(task_id=task_id).first()
        if label_task:
            self.process = []
            self.error_code = 1
            self.process.append(label_task.to_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        label_task = db.session.query(LabelProcess).filter_by(task_id=task_id).first()
        vs = label_task.status
        value = 0
        if vs == 'starting':
            label_task = db.session.query(LabelProcess).filter_by(task_id=task_id).first()
            label_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    label_task = db.session.query(LabelProcess).filter_by(task_id=task_id).first()
                    label_task.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        label_task = db.session.query(LabelProcess).filter_by(task_id=task_id).first()
                        label_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(label_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value