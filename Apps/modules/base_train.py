# -*- coding:utf-8 -*-
from Apps.models import TrainTask, Datas
from Apps import db, mach_id
from flask import current_app
import re
# from config import mach_id


class BaseTrainTask(object):

    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def create(self, task_id, task_name, network_path, task_describe, data_path, net_desc_id, data_desc_id, status,
               types, image_type, parallel_bool):
        traintasks = TrainTask()
        traintasks.task_id = task_id
        traintasks.task_name = task_name
        traintasks.network_path = network_path
        traintasks.task_describe = task_describe
        traintasks.data_path = data_path
        traintasks.net_desc_id = net_desc_id
        traintasks.data_desc_id = data_desc_id
        traintasks.status = status
        traintasks.machine_id = mach_id
        traintasks.type = types
        traintasks.image_type = image_type
        traintasks.parallel_bool = parallel_bool

        try:
            db.session.add(traintasks)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def start_training(self, task_id, category_num, iter_num, learning_rate, batch_size, steps, gpus, start_date,
                       status, model, weights):
        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        train_task.category_num = category_num
        train_task.iter_num = iter_num
        train_task.learning_rate = learning_rate
        train_task.batch_size = batch_size
        train_task.steps = steps
        train_task.gpus = gpus
        train_task.model = model
        train_task.start_date = start_date
        train_task.weights = weights
        train_task.status = status
        task_name = train_task.task_name
        network_path = train_task.network_path
        task_types = train_task.type
        data_desc_id = train_task.data_desc_id
        image_type = train_task.image_type
        parallel_bool = train_task.parallel_bool

        try:
            db.session.add(train_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        data = db.session.query(Datas).filter_by(id=data_desc_id).first()
        data_name = data.data_name
        return task_name, data_name, task_types, network_path, image_type, parallel_bool

    def end_train(self, task_id, end_date, status):
        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        train_task.status = status
        train_task.end_date = end_date
        task_name = train_task.task_name
        parallel_bool = train_task.parallel_bool
        try:
            db.session.add(train_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return task_name, parallel_bool

    def task_history(self):
        task_hitys = db.session.query(TrainTask).order_by(TrainTask.id.desc()).all()
        self.error_code = 1
        self.process = []
        for task_hity in task_hitys:
            self.process.append(task_hity.to_dict())

    def task_status(self, task_id):
        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        task_name = train_task.task_name
        task_type = train_task.type
        host = train_task.machine_id
        self.process = []
        self.error_code = 1
        self.process.append(train_task.to_dict())
        return task_name, task_type, host

    def callback(self, task_id, status, times):
        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        vs = train_task.status
        value = 0
        if vs == 'starting':
            train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
            train_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'completed!':
                    train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
                    train_task.status = status
                    train_task.end_date = times
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
                        train_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(train_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
