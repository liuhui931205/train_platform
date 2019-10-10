# -*-coding:utf-8-*-
from Apps.models import ScoreTasks
from Apps import db
from flask import current_app
import re


class BaseScoreTask(object):

    def start(self, task_id, area_name, scence_name,png_path, gpus, img, status, host):
        data = ScoreTasks()
        data.task_id = task_id
        data.area_name = area_name
        data.gpus = gpus
        data.png_path = png_path
        data.img = img
        data.scence_name = scence_name
        data.status = status
        data.host = host
        try:
            db.session.add(data)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def data_history(self):
        data_hitys = db.session.query(ScoreTasks).order_by(ScoreTasks.id.desc()).all()
        self.process = []
        for data_hity in data_hitys:
            self.process.append(data_hity.to_data_dict())

    def data_status(self, task_id):
        data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
        result = []
        if data_task:
            result.append(data_task.to_data_dict())
        return result

    def data_update(self, task_id, fields):
        data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
        for k, v in fields.items():
            setattr(data_task, k, v)
        try:
            db.session.add(data_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def callback(self, task_id, status):
        data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
        vs = data_task.status
        value = 0
        if vs == 'starting':
            data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
            data_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if status == 'unfinshed':
                    data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
                    data_task.status = status
                    value = 1
                else:
                    if (int(status) - result) > 5.0:
                        data_task = db.session.query(ScoreTasks).filter_by(task_id=task_id).first()
                        data_task.status = '%s%%' % status
                        value = 0
        try:
            db.session.add(data_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
