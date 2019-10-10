# -*- coding:utf-8 -*-
from Apps.models import LineDownload
from Apps import db
from flask import current_app
import re


class BaseLineDownTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, task_id, taskid_start, taskid_end, dest, status):
        line_download = LineDownload()
        line_download.taskid_start = taskid_start
        line_download.task_id = task_id
        line_download.taskid_end = taskid_end
        line_download.dest = dest
        line_download.status = status

        try:
            db.session.add(line_download)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def line_history(self):
        line_downloads = db.session.query(LineDownload).all()
        self.process = []
        for line_download in line_downloads:
            self.process.append(line_download.to_dict())

    def line_status(self, task_id):
        line_task = db.session.query(LineDownload).filter_by(task_id=task_id).first()
        if line_task:
            self.process = []
            self.error_code = 1
            self.process.append(line_task.to_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        line_task = db.session.query(LineDownload).filter_by(task_id=task_id).first()
        vs = line_task.status
        value = 0
        if status == 'completed!':
            line_task = db.session.query(LineDownload).filter_by(task_id=task_id).first()
            line_task.status = status
            value = 1
        elif vs == 'starting':
            line_task = db.session.query(LineDownload).filter_by(task_id=task_id).first()
            line_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if (int(status) - result) > 5.0:
                    line_task = db.session.query(LineDownload).filter_by(task_id=task_id).first()
                    line_task.status = '%s%%' % status
                    value = 0
        try:
            db.session.add(line_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
