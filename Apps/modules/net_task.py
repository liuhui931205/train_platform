# -*-coding:utf-8-*-
import os
from .base_network import BaseNetTask
from Apps.utils.copy_all import copyFiles


class NetTasks(BaseNetTask):
    def __init__(self):
        super(NetTasks, self).__init__()
        self.error_code = 1
        self.message = 'Network start'
        self.task_id = ''
        self.status = ''

    def create_network(self, task_id, net_name, src_network, net_describe, status):

        self.task_id = task_id
        network = '/data/deeplearning/train_platform/network/' + net_name
        if os.path.exists(network):
            self.error_code = 0
            self.message = 'network already exist'
        else:
            os.mkdir(network)
            copyFiles('/data/deeplearning/network_template/' + src_network, network)
            self.error_code = 1
            self.message = 'network create success'
            self.status = 'completed'
            self.start(task_id, net_name, src_network, net_describe, status)
            self.update_task()

    def update_task(self):
        self.callback(self.task_id, self.status)
