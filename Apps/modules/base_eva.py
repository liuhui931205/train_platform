# -*- coding:utf-8 -*-
from Apps.models import Evaluate_Models
from Apps import db
from flask import current_app
import re


class BaseEvaTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, task_id, sour_dir, gpus, dest_dir, single_gpu, model, status, host):
        eva_model = Evaluate_Models()
        eva_model.task_id = task_id
        sour = ""
        for i in sour_dir:
            sour += i + ','
        eva_model.sour_dir = sour
        eva_model.gpus = gpus
        eva_model.dest_dir = dest_dir
        eva_model.single_gpu = single_gpu
        eva_model.model = str(model)
        eva_model.status = status
        eva_model.host = host


        try:
            db.session.add(eva_model)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def eva_history(self):
        eva_hitys = db.session.query(Evaluate_Models).all()
        self.process = []
        self.error_code = 1
        for eva_hity in eva_hitys:
            self.process.append(eva_hity.to_model_dict())

    def eva_status(self, task_id):
        eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
        if eva_task:
            self.process = []
            self.error_code = 1
            self.process.append(eva_task.to_model_dict())
        else:
            self.error_code = 0
            self.process = []

    def eva_query(self, task_id):
        eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
        if eva_task:
            self.error_code = 1
            sour_dir = eva_task.sour_dir
            dest_dir = eva_task.dest_dir
            host = eva_task.host
            return sour_dir, dest_dir, host
        else:
            self.error_code = 0
            return

    def callback(self, task_id, status):
        eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
        vs = eva_task.status
        value = 0
        if vs == 'starting':
            eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
            eva_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
                    eva_task.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        eva_task = db.session.query(Evaluate_Models).filter_by(task_id=task_id).first()
                        eva_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(eva_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
