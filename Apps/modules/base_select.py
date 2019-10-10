# -*- coding:utf-8 -*-
import json
from Apps.models import Auto_select
from Apps import db
from flask import current_app
import re


class BaseSelectTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status, isshuffle, task_type, task_file):
        autoselect = Auto_select()
        autoselect.task_id = task_id
        autoselect.task_type = task_type
        autoselect.task_file = task_file
        autoselect.isshuffle = isshuffle
        autoselect.output_dir = output_dir
        autoselect.gpus = gpus
        autoselect.sele_ratio = sele_ratio
        autoselect.weights_dir = weights_dir
        autoselect.track_file = track_file
        autoselect.status = status
        try:
            db.session.add(autoselect)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def task_history(self):
        task_hitys = db.session.query(Auto_select).all()
        self.process = []
        for task_hity in task_hitys:
            self.process.append(task_hity.to_dict())
        return self.process

    def task_status(self, task_id):
        auto_select = db.session.query(Auto_select).filter_by(task_id=task_id).first()
        self.process = []
        self.error_code = 1
        self.process.append(auto_select.to_dict())

    def callback(self, task_id, status):
        auto_select = db.session.query(Auto_select).filter_by(task_id=task_id).first()
        vs = auto_select.status
        value = 0
        if vs == 'starting':
            auto_select = db.session.query(Auto_select).filter_by(task_id=task_id).first()
            auto_select.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    auto_select = db.session.query(Auto_select).filter_by(task_id=task_id).first()
                    auto_select.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        auto_select = db.session.query(Auto_select).filter_by(task_id=task_id).first()
                        auto_select.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(auto_select)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value

