# -*- coding:utf-8 -*-
from Apps.models import OfflineImport
from Apps import db
from flask import current_app
import re


class BaseOfflineTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, task_id, roadelement, source, author, annotype, datakind, city, imgoprange, status):
        off_import = OfflineImport()
        off_import.roadelement = roadelement
        off_import.task_id = task_id
        off_import.source = source
        off_import.author = author
        off_import.annotype = annotype
        off_import.datakind = datakind
        off_import.city = city
        off_import.imgoprange = imgoprange
        off_import.status = status
        try:
            db.session.add(off_import)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def off_history(self):
        off_hitys = db.session.query(OfflineImport).all()
        self.process = []
        for off_hity in off_hitys:
            self.process.append(off_hity.to_dict())

    def off_status(self, task_id):
        off_task = db.session.query(OfflineImport).filter_by(task_id=task_id).first()
        if off_task:
            self.process = []
            self.error_code = 1
            self.process.append(off_task.to_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        off_task = db.session.query(OfflineImport).filter_by(task_id=task_id).first()
        vs = off_task.status
        value = 0
        if status == 'completed!':
            off_task = db.session.query(OfflineImport).filter_by(task_id=task_id).first()
            off_task.status = status
            value = 1
        elif vs == 'starting':
            off_task = db.session.query(OfflineImport).filter_by(task_id=task_id).first()
            off_task.status = '0%'
            value = 0
        else:
            result = re.match(r'\d+', vs)
            if result:
                result = int(result.group())
                if (int(status) - result) > 5.0:
                    off_task = db.session.query(OfflineImport).filter_by(task_id=task_id).first()
                    off_task.status = '%s%%' % status
                    value = 0
        try:
            db.session.add(off_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
        return value
