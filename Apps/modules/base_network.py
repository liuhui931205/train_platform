# -*- coding:utf-8 -*-
from Apps.models import NetworkDescribe
from Apps import db
from flask import current_app
import re


class BaseNetTask(object):
    def __init__(self):
        self.error_code = 1
        self.process = []
        self.message = "start a job"
        self._callback = ""

    def start(self, task_id, net_name, src_network, net_describe, status):
        network = NetworkDescribe()
        network.task_id = task_id
        network.net_name = net_name
        network.src_network = src_network
        network.net_describe = net_describe
        network.status = status
        try:
            db.session.add(network)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)

    def net_history(self):
        net_task = db.session.query(NetworkDescribe).order_by(NetworkDescribe.id.desc()).all()
        self.process = []
        self.error_code = 1
        for net_hity in net_task:
            self.process.append(net_hity.to_net_dict())
        return self.process

    def net_status(self, task_id):
        net_task = db.session.query(NetworkDescribe).filter_by(task_id=task_id).first()

        self.process = []
        if net_task:
            self.error_code = 1
            self.process.append(net_task.to_net_dict())
        else:
            self.error_code = 0
            self.process = []

    def callback(self, task_id, status):
        net_task = db.session.query(NetworkDescribe).filter_by(task_id=task_id).first()
        net_task.status = status

        try:
            db.session.add(net_task)
            db.session.commit()
        except Exception as e:
            current_app.logger.error(e)
