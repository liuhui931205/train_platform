# -*-coding:utf-8-*-
import json
import scp
import multiprocessing
from Apps.models import NetworkDescribe, Datas, TrainTask
from Apps import db, mach_id
from flask import current_app
from Apps.utils.copy_all import copyFiles
import time
from .base_train import BaseTrainTask
import re
import os
import shutil
from Apps.utils.client import client, create_ssh_client
import requests
from Apps.utils.parallel_train import upload_datas, upload_folder, rm_datas, rm_folder
from Apps.libs.s_eva_info.eval import EvalData
from config import url_mach, data0_dest_data, data0_sour_train, data_dest_data, data_sour_train, parallel_training


class TrainTasks(BaseTrainTask):

    def __init__(self):
        super(TrainTasks, self).__init__()
        self.error_code = 0
        self.message = 'Autosele start'
        self.task_id = ''
        self.status = ''

    def create_task(self, task_name, network, data, task_desc, task_id, status, taskid, types, image_type,
                    parallel_bool, map_template):
        self.task_id = task_id
        train_path = data_sour_train
        t_path = os.path.join(train_path, task_name)
        if task_name:
            if os.path.exists(t_path):
                self.error_code = 0
                self.message = 'task already exist!'
            else:
                network_path = os.path.join('/data/deeplearning/train_platform/network', network)
                data_path = os.path.join(data_dest_data, data)

                net_desc = db.session.query(NetworkDescribe).filter_by(net_name=network).first()
                net_desc_id = net_desc.id
                data_desc = db.session.query(Datas).filter_by(data_name=data).first()

                data_desc_id = data_desc.id
                # 创建训练任务文件夹
                os.mkdir(os.path.join(train_path, task_name))
                train_path = os.path.join(train_path, task_name)
                t_net_path = os.path.join(train_path, 'network')
                os.mkdir(t_net_path)
                # 拷贝models和conf
                if types == "semantic-segmentation":
                    if network == "s_network":
                        train_start_path = os.path.join('/data/deeplearning/train_platform/task_template', 'seg_start')
                    else:
                        train_start_path = os.path.join('/data/deeplearning/train_platform/task_template',
                                                        'seg_start_up')
                elif types == "classification-model":
                    train_start_path = os.path.join('/data/deeplearning/train_platform/task_template', 'cls_start')
                elif types == "target-detection":
                    train_start_path = os.path.join('/data/deeplearning/train_platform/task_template', 'tar_start')
                copyFiles(train_start_path, train_path)
                if types == "semantic-segmentation":
                    shutil.copyfile(
                        os.path.join("/data/deeplearning/train_platform/task_template/maps_templates", map_template,
                                     "label_map.txt"), os.path.join(train_path, "conf/label_map.txt"))
                if taskid:
                    train_task = db.session.query(TrainTask).filter_by(task_id=taskid).first()
                    t_id = train_task.machine_id
                    t_name = train_task.task_name
                    # model_path = '/data/deeplearning/train_platform/train_task/' + t_name + '/output/models'
                    model_path = os.path.join(data0_sour_train, t_name, 'output/models')
                    # conf_path = '/data/deeplearning/train_platform/train_task/' + t_name + '/conf/label_map.txt'
                    # if t_id != mach_id:
                    dest_scp, dest_sftp, dest_ssh = client(id="host")
                    li = dest_sftp.listdir(model_path)
                    model_li = []
                    for i in li:
                        if i.endswith('.json'):
                            if not os.path.exists(
                                    os.path.join(
                                        '/data/deeplearning/train_platform/train_task/' + task_name + '/models', i)):
                                dest_sftp.get(
                                    os.path.join(model_path, i),
                                    os.path.join(
                                        '/data/deeplearning/train_platform/train_task/' + task_name + '/models', i))
                        elif i.endswith('.params'):
                            model_li.append(i)
                    model_li.sort()
                    get_path = model_li[-1][:-11] + '0000.params'
                    if not os.path.exists(
                            os.path.join('/data/deeplearning/train_platform/train_task/' + task_name + '/models',
                                         get_path)):
                        dest_sftp.get(
                            os.path.join(model_path, model_li[-1]),
                            os.path.join('/data/deeplearning/train_platform/train_task/' + task_name + '/models',
                                         get_path))
                    # dest_sftp.get(conf_path,'/data/deeplearning/train_platform/train_task/' + task_name + '/conf/label_map.txt')
                    dest_sftp.close()
                    dest_scp.close()
                    dest_ssh.close()
                if not os.path.exists(data_path):
                    os.makedirs(data_path)
                    dest_scp, dest_sftp, dest_ssh = client(id="host")
                    data_li = dest_sftp.listdir(os.path.join(data0_dest_data, data))
                    for j in data_li:
                        path = os.path.join(data_path, j)
                        if not os.path.exists(path):
                            dest_sftp.get(os.path.join(data0_dest_data, data, j), path)
                    dest_sftp.close()
                    dest_scp.close()
                    dest_ssh.close()

                    # copyFiles(os.path.join(data0_dest_data, data), data_path)
                self.create(task_id, task_name, network_path, task_desc, data_path, net_desc_id, data_desc_id, status,
                            types, image_type, parallel_bool)
                self.error_code = 1
                self.message = 'create task success'

    def start_task(self, task_id, data_type, category_num, gpus, batch_size, weights, iter_num, model, steps,
                   learning_rate, start_date, train_con, data_con, status):

        task_name, data_name, task_types, network_path, image_type, parallel_bool = self.start_training(
            task_id, category_num, iter_num, learning_rate, batch_size, steps, gpus, start_date, status, model,
            weights)
        if int(parallel_bool) and task_types == 'semantic-segmentation':

            train_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/conf/seg_train.json'
            data_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/conf/seg_train_data.json'
            train_con = json.loads(train_con)
            data_con = json.loads(data_con)
            json.dump(train_con, open(train_path, 'w'))
            json.dump(data_con, open(data_path, 'w'))
            upload_datas(data_name)
            upload_folder(task_name)
            pro = multiprocessing.Process(target=self.p_task_async,
                                          args=(task_name, weights, data_type, data_name, model, category_num,
                                                batch_size, steps, iter_num, learning_rate, gpus, network_path))
            pro.start()
            time.sleep(5)
            log_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/log/' + model + '.train.log.' + data_name

            pros = multiprocessing.Process(target=self.p_update_task,
                                           args=(task_id, log_path, iter_num, task_name, data_name))
            pros.start()
            self.error_code = 1
            self.message = 'traintask start success'
        else:
            if task_types == 'semantic-segmentation':
                train_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/conf/seg_train.json'
                data_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/conf/seg_train_data.json'
                train_con = json.loads(train_con)
                data_con = json.loads(data_con)
                json.dump(train_con, open(train_path, 'w'))
                json.dump(data_con, open(data_path, 'w'))
                # cmd_info = self.task_async(task_name, weights, data_type, data_name, model, category_num, batch_size, steps,
                #                            iter_num,
                #                            learning_rate, gpus, task_types, network_path)
                pro = multiprocessing.Process(target=self.task_async,
                                              args=(task_name, weights, data_type, data_name, model, category_num,
                                                    batch_size, steps, iter_num, learning_rate, gpus, task_types,
                                                    network_path))
                pro.start()
                log_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/log/' + model + '.train.log.' + data_name
                time.sleep(10)
                pros = multiprocessing.Process(target=self.update_task,
                                               args=(task_id, task_types, log_path, iter_num, task_name, gpus,
                                                     category_num, data_type, data_name, image_type))
                pros.start()
            elif task_types == 'classification-model':
                pro = multiprocessing.Process(target=self.task_async,
                                              args=(task_name, weights, data_type, data_name, model, category_num,
                                                    batch_size, steps, iter_num, learning_rate, gpus, task_types,
                                                    network_path))
                pro.start()
                log_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/log/' + model + '.train.log.' + data_name
                time.sleep(10)
                pros = multiprocessing.Process(target=self.update_task,
                                               args=(task_id, task_types, log_path, iter_num, task_name, gpus,
                                                     category_num, data_type))
                pros.start()
            elif task_types == 'target-detection':
                if model in ["KuandengSign", "KuandengPole", "Kuandeng"]:
                    pro = multiprocessing.Process(target=self.task_async,
                                                  args=(task_name, weights, data_type, data_name, model, category_num,
                                                        batch_size, steps, iter_num, learning_rate, gpus, task_types,
                                                        network_path))
                    pro.start()
                    log_path = '/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{}'.format(
                        task_name, model, data_name)

                    time.sleep(10)
                    pros = multiprocessing.Process(target=self.update_task,
                                                   args=(task_id, task_types, log_path, iter_num, task_name, gpus,
                                                         category_num, data_type, data_name, image_type))
                    pros.start()
                else:
                    self.error_code = 0
                    self.message = 'traintask start failed'
                    return

            try:
                process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                if process >= 1:
                    self.error_code = 1
                    self.message = 'traintask start success'
                else:
                    self.error_code = 0
                    self.message = 'traintask start failed'
            except Exception as e:
                current_app.logger.error(e)

    def end_task(self, task_id, end_date, status):
        self.task_id = task_id
        task_name, parallel_bool = self.end_train(task_id, end_date, status)
        if int(parallel_bool):
            dress = parallel_training[0]
            dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                         dress["dest_scp_passwd"])
            cmd_info = "bash /data/deeplearning/train_platform/network/stop.sh"
            for i in range(3):
                dest_ssh.exec_command(cmd_info, timeout=50)
            self.error_code = 1
            self.message = 'task closed success'
        else:
            try:
                os.system("ps aux | grep -w '" + task_name + "' |grep -v grep| cut -c 9-15 | xargs kill -9")
                self.error_code = 1
                self.message = 'task closed success'
            except Exception as e:
                current_app.logger.error(e)
                self.error_code = 0
                self.message = 'task closed fail'

    def con_task(self, task_id):
        models_list = []
        train_task = db.session.query(TrainTask).filter_by(task_id=task_id).first()
        if train_task.status != "completed!":
            machine_id = train_task.machine_id
            category_num = train_task.category_num
            image_type = train_task.image_type
            task_name = train_task.task_name
            weights = train_task.weights
            iter_num = train_task.iter_num
            learning_rate = train_task.learning_rate
            batch_size = train_task.batch_size
            network_path = train_task.network_path
            steps = train_task.steps
            gpus = train_task.gpus
            model = train_task.model
            task_types = train_task.type
            parallel_bool = train_task.parallel_bool
            data_desc_id = train_task.data_desc_id
            data = db.session.query(Datas).filter_by(id=data_desc_id).first()
            data_name = data.data_name
            data_type = data.data_type
            mod_path = os.path.join(data_sour_train, task_name, 'output/models')
            if task_types == "semantic-segmentation":
                if machine_id != mach_id:
                    url = url_mach[machine_id] + "continue_task"
                    data = {'task_id': task_id}
                    json_datas = requests.post(url=url, json=data)
                    js_data = json.loads(json_datas.text)
                    if js_data['errno'] == 1 and js_data['message'] == 'continue task success':
                        self.error_code = 1
                        self.message = 'continue task success'
                    else:
                        self.error_code = 1
                        self.message = 'task already working'
                else:
                    if int(parallel_bool):
                        # log_path = '/data/deeplearning/train_platform/train_task/' + task_name + '/log/' + model + '.train.log.' + data_name
                        log_path = os.path.join(data_sour_train, task_name, 'log', model + '.train.log.' + data_name)
                        dress = parallel_training[0]
                        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"],
                                                     dress["dest_scp_user"], dress["dest_scp_passwd"])
                        dest_scp = scp.SCPClient(dest_ssh.get_transport())
                        dest_sftp = dest_ssh.open_sftp()

                        path = os.path.join(data_sour_train, task_name, "output/models")

                        files = dest_sftp.listdir(path)
                        index = len(files) - 3
                        dest_ssh.close()
                        dest_scp.close()
                        dest_sftp.close()

                        if index > 0:
                            weights = "%s_ep-%04d.params" % (model, index * 5)
                            # weights = '/data/deeplearning/train_platform/train_task/' + task_name + '/output/models/' + weights
                            weights = os.path.join(data_sour_train, task_name, 'output/models/', weights)
                            cmd_info = "/home/kdreg/src/apache-mxnet-src-1.2.1-incubating/tools/launch.py -n {} --launcher ssh -H /data/deeplearning/train_platform/network/hosts 'export PYTHONPATH=/data/deeplearning/train_platform/network/s_network/segmentation:$PYTHONPATH && export DMLC_INTERFACE=bond0 && ulimit -c unlimited && python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights={} --num-epoch={} --steps={} --base-lr={}' >/dev/null 2>&1".format(
                                len(parallel_training), network_path, task_name, data_name, data_type, task_name,
                                model, task_name, model, data_name, category_num, gpus, batch_size, weights, iter_num,
                                steps, learning_rate)
                            dress = parallel_training[0]
                            dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"],
                                                         dress["dest_scp_user"], dress["dest_scp_passwd"])
                            dest_ssh.exec_command(cmd_info)
                            dest_ssh.close()
                            pros = multiprocessing.Process(target=self.p_update_task,
                                                           args=(task_id, log_path, iter_num, task_name, data_name))
                            pros.start()
                            self.error_code = 1
                            self.message = 'continue task success'
                        else:
                            # weights = '/data/deeplearning/train_platform/train_task/' + task_name + '/output/models/' + weights
                            weights = os.path.join(data_sour_train, task_name, 'models/', weights)
                            cmd_info = "/home/kdreg/src/apache-mxnet-src-1.2.1-incubating/tools/launch.py -n {} --launcher ssh -H /data/deeplearning/train_platform/network/hosts 'export PYTHONPATH=/data/deeplearning/train_platform/network/s_network/segmentation:$PYTHONPATH && export PS_VERBOSE=1 && export DMLC_INTERFACE=bond0 && ulimit -c unlimited && python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights={} --num-epoch={} --steps={} --base-lr={}' >/dev/null 2>&1".format(
                                len(parallel_training), network_path, task_name, data_name, data_type, task_name,
                                model, task_name, model, data_name, category_num, gpus, batch_size, weights, iter_num,
                                steps, learning_rate)
                            dress = parallel_training[0]
                            dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"],
                                                         dress["dest_scp_user"], dress["dest_scp_passwd"])
                            dest_ssh.exec_command(cmd_info)
                            dest_ssh.close()
                            pros = multiprocessing.Process(target=self.p_update_task,
                                                           args=(task_id, log_path, iter_num, task_name, data_name))
                            pros.start()
                            self.error_code = 1
                            self.message = 'continue task success'

                    else:
                        mod_li = os.listdir(mod_path)
                        for i in mod_li:
                            if i.endswith('.params'):
                                models_list.append(i)
                        if models_list:
                            models_list.sort()
                            weights = os.path.join(mod_path, models_list[-1])

                        else:
                            weights = os.path.join(data_sour_train, task_name, 'models/', weights)
                        cmd_info = 'nohup python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights={} --num-epoch={} --steps={} --base-lr={} &'.format(
                            network_path, task_name, data_name, data_type, task_name, model, task_name, model,
                            data_name, category_num, gpus, batch_size, weights, iter_num, steps, learning_rate)
                        process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                        if not process:
                            os.system(cmd_info)
                            time.sleep(5)
                            log_path = os.path.join(data_sour_train, task_name, 'log',
                                                    model + '.train.log.' + data_name)
                            pros = multiprocessing.Process(target=self.update_task,
                                                           args=(task_id, task_types, log_path, iter_num, task_name,
                                                                 gpus, category_num, data_type, data_name, image_type))
                            pros.start()
                            self.error_code = 1
                            self.message = 'continue task success'
                        else:
                            self.error_code = 1
                            self.message = 'task already working'
            elif task_types == "target-detection":
                if machine_id != mach_id:
                    url = url_mach[machine_id] + "continue_task"
                    data = {'task_id': task_id}
                    json_datas = requests.post(url=url, json=data)
                    js_data = json.loads(json_datas.text)
                    if js_data['errno'] == 1 and js_data['message'] == 'continue task success':
                        self.error_code = 1
                        self.message = 'continue task success'
                    else:
                        self.error_code = 1
                        self.message = 'task already working'
                else:
                    iter_num = iter_num.split(",")
                    learning_rate = learning_rate.split(",")
                    steps = steps.split("_")
                    rpn_epoch = int(iter_num[0])
                    rpn_lr = float(learning_rate[0])
                    rpn_lr_step = steps[0]
                    rcnn_epoch = int(iter_num[1])
                    rcnn_lr = float(learning_rate[1])
                    rcnn_lr_step = steps[1]
                    log_path = os.path.join(data_sour_train, task_name, 'log', model + '.train.log.' + data_name)
                    resume = self.re_log(log_path)
                    mod_li = os.listdir(mod_path)
                    for i in mod_li:
                        if int(resume) == 1 and i.startswith("rpn1") and i.endswith(".params"):
                            models_list.append(i)
                        elif int(resume) == 3 and i.startswith("rcnn1") and i.endswith(".params"):
                            models_list.append(i)
                        elif int(resume) == 4 and i.startswith("rpn2") and i.endswith(".params"):
                            models_list.append(i)
                        elif int(resume) == 6 and i.startswith("rcnn2") and i.endswith(".params"):
                            models_list.append(i)
                    if models_list:
                        models_list.sort()
                        begin_epoch = models_list[-1][-11:-7]
                        cmd_info = "bash /data/deeplearning/train_platform/network/mx-maskrcnn/train.sh -n resnet_fpn -d {} -D /data/deeplearning/train_platform/data/{} -p /data/deeplearning/train_platform/network/mx-maskrcnn/model/resnet-50 -P /data/deeplearning/train_platform/train_task/{} -o /data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} -c /data/deeplearning/train_platform/train_task/{}/conf/tar_train.json -e {} -l {} -s {} -E {} -L {} -S {} -g {} -b {} -R {}".format(
                            model, data_name, task_name, task_name, model, data_name, task_name, rpn_epoch, rpn_lr,
                            rpn_lr_step, rcnn_epoch, rcnn_lr, rcnn_lr_step, gpus, begin_epoch, resume)
                    else:
                        cmd_info = "bash /data/deeplearning/train_platform/network/mx-maskrcnn/train.sh -n resnet_fpn -d {} -D /data/deeplearning/train_platform/data/{} -p /data/deeplearning/train_platform/network/mx-maskrcnn/model/resnet-50 -P /data/deeplearning/train_platform/train_task/{} -o /data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} -c /data/deeplearning/train_platform/train_task/{}/conf/tar_train.json -e {} -l {} -s {} -E {} -L {} -S {} -g {} -R {}".format(
                            model, data_name, task_name, task_name, model, data_name, task_name, rpn_epoch, rpn_lr,
                            rpn_lr_step, rcnn_epoch, rcnn_lr, rcnn_lr_step, gpus, resume)

                    process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                    if not process:
                        os.system(cmd_info)
                        time.sleep(5)
                        log_path = os.path.join(data_sour_train, task_name, 'log', model + '.train.log.' + data_name)
                        pros = multiprocessing.Process(target=self.update_task,
                                                       args=(task_id, task_types, log_path, iter_num, task_name, gpus,
                                                             category_num, data_type, data_name, image_type))
                        pros.start()
                        self.error_code = 1
                        self.message = 'continue task success'
                    else:
                        self.error_code = 1
                        self.message = 'task already working'

        else:
            self.error_code = 1
            self.message = 'task already completed!'

    def task_async(self, task_name, weights, data_type, data_name, model, category_num, batch_size, steps, iter_num,
                   learning_rate, gpus, task_types, network_path):
        if task_types == "semantic-segmentation":
            if weights:
                weights = '/data/deeplearning/train_platform/train_task/' + task_name + '/models/' + weights
                cmd_info = 'python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights={} --num-epoch={} --steps={} --base-lr={}'.format(
                    network_path, task_name, data_name, data_type, task_name, model, task_name, model, data_name,
                    category_num, gpus, batch_size, weights, iter_num, steps, learning_rate)

            else:
                cmd_info = 'python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights=/data/deeplearning/train_platform/train_task/{}/models/cityscapes_rna-a1_cls19_s8_ep-0001.params --num-epoch={} --steps={} --base-lr={}'.format(
                    network_path, task_name, data_name, data_type, task_name, model, task_name, model, data_name,
                    category_num, gpus, batch_size, task_name, iter_num, steps, learning_rate)
        elif task_types == 'classification-model':
            data_path = "/data/deeplearning/train_platform/data/" + data_name + "/" + data_name + "_train.idx"
            with open(data_path, 'r') as f:
                data = f.readlines()
            num_examples = len(data)
            cmd_info = 'nohup python2 -u {}/train_densenet_wd.py --train_prefix=/data/deeplearning/train_platform/data/{}/{}_train --val=/data/deeplearning/train_platform/data/{}/{}_val --model_prefix=/data/deeplearning/train_platform/train_task/{}/output/models/{} --log=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --step={} --lr={} --depth=169 --num-examples={} &'.format(
                network_path, data_name, data_name, data_name, data_name, task_name, model, task_name, model,
                data_name, category_num, gpus, batch_size, steps, learning_rate, num_examples)
        elif task_types == 'target-detection':
            iter_num = iter_num.split(",")
            learning_rate = learning_rate.split(",")
            steps = steps.split("_")
            rpn_epoch = int(iter_num[0])
            rpn_lr = float(learning_rate[0])
            rpn_lr_step = steps[0]
            rcnn_epoch = int(iter_num[1])
            rcnn_lr = float(learning_rate[1])
            rcnn_lr_step = steps[1]
            if weights:
                weights = "model/resnet-50"
                cmd_info = "bash /data/deeplearning/train_platform/network/mx-maskrcnn/train.sh -n resnet_fpn -d {} -D /data/deeplearning/train_platform/data/{} -p /data/deeplearning/train_platform/network/mx-maskrcnn/{} -P /data/deeplearning/train_platform/train_task/{} -o /data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} -c /data/deeplearning/train_platform/train_task/{}/conf/tar_train.json -e {} -l {} -s {} -E {} -L {} -S {} -g {}".format(
                    model, data_name, weights, task_name, task_name, model, data_name, task_name, rpn_epoch, rpn_lr,
                    rpn_lr_step, rcnn_epoch, rcnn_lr, rcnn_lr_step, gpus)
            else:
                cmd_info = "bash /data/deeplearning/train_platform/network/mx-maskrcnn/train.sh -n resnet_fpn -d {} -D /data/deeplearning/train_platform/data/{} -p /data/deeplearning/train_platform/network/mx-maskrcnn/model/resnet-50 -P /data/deeplearning/train_platform/train_task/{} -o /data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} -c /data/deeplearning/train_platform/train_task/{}/conf/tar_train.json -e {} -l {} -s {} -E {} -L {} -S {} -g {}".format(
                    model, data_name, task_name, task_name, model, data_name, task_name, rpn_epoch, rpn_lr,
                    rpn_lr_step, rcnn_epoch, rcnn_lr, rcnn_lr_step, gpus)
        os.system(cmd_info)

    def update_task(self, task_id, task_types, path, iter_num, task_name, gpus, category_num, data_type, data_name,
                    image_type):
        if task_types == "semantic-segmentation":
            while True:
                p_id = multiprocessing.current_process().pid
                current_app.logger.info(p_id)
                time.sleep(180)
                process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                if process >= 1:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        lis = []
                        nd = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"base_module.*Train-mean-iou=\d+\.\d+", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            for j in li:
                                n = j[0].split('=')
                                lis.append(n[-1])
                                d = re.findall(r'\d+', n[0])
                                nd.append(d[-1])
                            if int(nd[-1]) != int(iter_num) - 1:
                                times = ""
                                self.status = (int(nd[-1]) / ((int(iter_num) - 1) * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                self.status = 'completed!'
                                time.sleep(180)
                                times = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                                value = self.callback(task_id, self.status, times)
                            if value:
                                self.rm_model(task_types, task_name, gpus, category_num, data_type, data_name,
                                              image_type)
                                break
                    except Exception as e:
                        current_app.logger.error(e)
                else:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        lis = []
                        nd = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"base_module.*Train-mean-iou=\d+\.\d+", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            for j in li:
                                n = j[0].split('=')
                                lis.append(n[-1])
                                d = re.findall(r'\d+', n[0])
                                nd.append(d[-1])
                            if int(nd[-1]) != int(iter_num) - 1:
                                times = ""
                                self.status = (int(nd[-1]) / ((int(iter_num) - 1) * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                self.status = 'completed!'
                                time.sleep(180)
                                times = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                value = self.callback(task_id, self.status, times)
                            if value:
                                self.rm_model(task_types, task_name, gpus, category_num, data_type, data_name,
                                              image_type)
                                break
                    except Exception as e:
                        current_app.logger.error(e)
                        break
        elif task_types == "classification-model":
            while True:
                p_id = multiprocessing.current_process().pid
                current_app.logger.info(p_id)
                time.sleep(180)
                process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                if process >= 1:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        lis = []
                        nd = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"Epoch\[\d+\] Validation-accuracy=\d+\.\d+", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            for j in li:
                                n = j[0].split('=')
                                lis.append(n[-1])
                                d = re.findall(r'\d+', n[0])
                                nd.append(d[-1])
                            if int(nd[-1]) != int(iter_num) - 1:
                                times = ""
                                self.status = (int(nd[-1]) / ((int(iter_num) - 1) * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                self.status = 'completed!'
                                times = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                value = self.callback(task_id, self.status, times)
                            if value:
                                break
                    except Exception as e:
                        current_app.logger.error(e)
                else:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        lis = []
                        nd = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"Epoch\[\d+\] Validation-accuracy=\d+\.\d+", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            for j in li:
                                n = j[0].split('=')
                                lis.append(n[-1])
                                d = re.findall(r'\d+', n[0])
                                nd.append(d[-1])
                            if int(nd[-1]) != int(iter_num) - 1:
                                times = ""
                                self.status = (int(nd[-1]) / ((int(iter_num) - 1) * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                self.status = 'completed!'
                                times = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                                value = self.callback(task_id, self.status, times)
                            if value:
                                break
                    except Exception as e:
                        current_app.logger.error(e)
                        break
        elif task_types == "target-detection":
            while True:
                p_id = multiprocessing.current_process().pid
                current_app.logger.info(p_id)
                time.sleep(180)
                process = len(os.popen('ps aux | grep -w "' + task_name + '" | grep -v grep').readlines())
                if process >= 1:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"STEP:\d", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            resume = li[-1][0].split(":")[-1].strip()
                            if int(resume) != 6:
                                times = ""
                                self.status = (int(resume) / (6 * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                if os.path.exists(
                                        os.path.join(data_sour_train, task_name, "output", "models",
                                                     "final-0000.params")):
                                    self.status = 'completed!'
                                    time.sleep(180)
                                    times = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                                    value = self.callback(task_id, self.status, times)
                            if value:
                                self.rm_model(task_types, task_name, gpus, category_num, data_type, data_name,
                                              image_type)
                                break
                    except Exception as e:
                        current_app.logger.error(e)
                else:
                    try:
                        with open(path, 'r') as f:
                            data = f.readlines()
                        li = []
                        for i in data:
                            # m = re.findall(r'base_module.*Epoch\[\d+\]', i)
                            m = re.findall(r"STEP:\d", i)
                            if m:
                                li.append(m)
                        if len(li) > 0:
                            resume = li[-1][0].split(":")[-1].strip()
                            if int(resume) != 6:
                                times = ""
                                self.status = (int(resume) / (6 * 1.00)) * 100
                                value = self.callback(task_id, self.status, times)
                            else:
                                if os.path.exists(
                                        os.path.join(data_sour_train, task_name, "output", "models",
                                                     "final-0000.params")):
                                    self.status = 'completed!'
                                    time.sleep(180)
                                    times = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                                    value = self.callback(task_id, self.status, times)
                            if value:
                                self.rm_model(task_types, task_name, gpus, category_num, data_type, data_name,
                                              image_type)
                                break

                    except Exception as e:
                        current_app.logger.error(e)
                        break

    def rm_model(self, task_types, task_name, gpus, category_num, data_type, data_name, image_type):
        # paths = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/models"
        paths = os.path.join(data_sour_train, task_name, "output/models")
        model_li = os.listdir(paths)
        models = []
        for i in model_li:
            if i.endswith(".params"):
                models.append(i)
        models.sort()
        if task_types == "semantic-segmentation":
            dest_scp, dest_sftp, dest_ssh = client(id="host")
            map_path = os.path.join(data_sour_train, task_name, "conf/label_map.txt")
            eva_model = os.path.join(paths, models[-1])
            output_dir = os.path.join(data_sour_train, task_name, "output/tracker")
            model_desc_file = os.path.join(data_sour_train, task_name, "output/models/info.txt")
            record_list_file = os.path.join(data_dest_data, data_name, 'kd_' + data_type + '_test.lst')
            gpu_id = int(gpus.split(',')[0])
            cls = int(category_num)
            try:
                for line in file(record_list_file):
                    line = line.strip().split("\t")
                    dirs = os.path.dirname(line[2])
                    if not os.path.exists(dirs):
                        os.makedirs(dirs)
                    dest_sftp.get(line[2],line[2])
                    dest_sftp.get(line[3],line[3])
                label_map_data = []
                with open(map_path, 'r') as f:
                    while True:
                        line = f.readline()
                        if line:
                            line = (line.strip()).split('\t')
                            label_map_data.append(line)
                        else:
                            label_map_data.pop(0)
                            break
                with open(os.path.join(paths, 'map_label.json'), 'r') as d:
                    label_data = json.loads(d.read())

                eval_data = EvalData(eva_model,
                                     record_list_file,
                                     output_dir,
                                     model_desc_file,
                                     gpu_id,
                                     cls,
                                     image_type,
                                     use_half_image=True,
                                     save_result_or_not=False,
                                     iou_thresh_low=0.1,
                                     min_pixel_num=2000,
                                     flip=False,
                                     label_map_data=label_map_data,
                                     label_data=label_data)
                eval_data.run()
            except Exception as e:
                current_app.logger.error(e)
            finally:

                models.remove(models[-1])
                for i in models:
                    os.remove(os.path.join(paths, i))

                task_lists = dest_sftp.listdir(data0_sour_train)
                if task_name not in task_lists:
                    # dest_sftp.mkdir(os.path.join("/data/deeplearning/train_platform/train_task/", task_name))
                    # copyFiles(os.path.join(data_sour_train, task_name), os.path.join(data0_sour_train, task_name))
                    dest_scp.put(os.path.join(data_sour_train, task_name),
                                 os.path.join(data0_sour_train, task_name),
                                 recursive=True)
                dest_ssh.close()
                dest_scp.close()
                dest_sftp.close()

        elif task_types == "target-detection":
            for model in models:
                if model != "final-0000.params":
                    os.remove(os.path.join(paths, model))
            dest_scp, dest_sftp, dest_ssh = client(id="host")
            task_lists = dest_sftp.listdir(data0_sour_train)
            if task_name not in task_lists:
                # dest_sftp.mkdir(os.path.join("/data/deeplearning/train_platform/train_task/", task_name))
                # copyFiles(os.path.join(data_sour_train, task_name), os.path.join(data0_sour_train, task_name))
                dest_scp.put(os.path.join(data_sour_train, task_name),
                             os.path.join(data0_sour_train, task_name),
                             recursive=True)

    def re_log(self, log_path):
        with open(log_path, 'r') as f:
            data = f.readlines()
        li = []
        for i in data:
            m = re.findall(r"STEP:\d", i)
            if m:
                li.append(m)
        resume = li[-1][0].split(":")[-1].strip()
        return resume

    # 并行训练
    def p_task_async(self, task_name, weights, data_type, data_name, model, category_num, batch_size, steps, iter_num,
                     learning_rate, gpus, network_path):
        if weights:
            # weights = '/data/deeplearning/train_platform/train_task/' + task_name + '/models/' + weights
            weights = os.path.join(data_sour_train, task_name, 'models', weights)
            cmd_info = "/home/kdreg/src/apache-mxnet-src-1.2.1-incubating/tools/launch.py -n {} --launcher ssh -H /data/deeplearning/train_platform/network/hosts 'export PYTHONPATH=/data/deeplearning/train_platform/network/s_network/segmentation:$PYTHONPATH && export DMLC_INTERFACE=bond0 && ulimit -c unlimited && python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights={} --num-epoch={} --steps={} --base-lr={}'>/dev/null 2>&1".format(
                len(parallel_training), network_path, task_name, data_name, data_type, task_name, model, task_name,
                model, data_name, category_num, gpus, batch_size, weights, iter_num, steps, learning_rate)

        else:
            cmd_info = "/home/kdreg/src/apache-mxnet-src-1.2.1-incubating/tools/launch.py -n {} --launcher ssh -H /data/deeplearning/train_platform/network/hosts 'export PYTHONPATH=/data/deeplearning/train_platform/network/s_network/segmentation:$PYTHONPATH && export DMLC_INTERFACE=bond0 && ulimit -c unlimited && python2 {}/train.py --conf-file=/data/deeplearning/train_platform/train_task/{}/conf/seg_train.json --data-prefix=/data/deeplearning/train_platform/data/{}/kd_{} --output=/data/deeplearning/train_platform/train_task/{}/output --model={} --log-file=/data/deeplearning/train_platform/train_task/{}/log/{}.train.log.{} --num-classes={} --gpus={} --batch-size={} --weights=/data/deeplearning/train_platform/train_task/{}/models/cityscapes_rna-a1_cls19_s8_ep-0001.params --num-epoch={} --steps={} --base-lr={}'>/dev/null 2>&1".format(
                len(parallel_training), network_path, task_name, data_name, data_type, task_name, model, task_name,
                model, data_name, category_num, gpus, batch_size, task_name, iter_num, steps, learning_rate)
        dress = parallel_training[0]
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_ssh.exec_command(cmd_info)
        dest_ssh.close()

    # 并行更新
    def p_update_task(self, task_id, log_path, iter_num, task_name, data_name):
        while True:
            try:
                time.sleep(180)
                dress = parallel_training[0]
                dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                             dress["dest_scp_passwd"])
                dest_scp = scp.SCPClient(dest_ssh.get_transport())
                dest_sftp = dest_ssh.open_sftp()
                path = os.path.join(data_sour_train, task_name, "log")
                log_name = os.path.basename(log_path)
                files = dest_sftp.listdir(path)
                if log_name in files:
                    dest_sftp.get(log_path, log_path)
                    with open(log_path, 'r') as f:
                        log_data = f.readlines()
                    li = []
                    lis = []
                    nd = []
                    for line in log_data:
                        m = re.findall(r"base_module.*Train-mean-iou=\d+\.\d+", line)
                        if m:
                            li.append(m)
                    if li:
                        for j in li:
                            n = j[0].split('=')
                            lis.append(n[-1])
                            d = re.findall(r'\d+', n[0])
                            nd.append(d[-1])
                        if int(nd[-1]) != int(iter_num) - 1:
                            times = ""
                            self.status = (int(nd[-1]) / ((int(iter_num) - 1) * 1.00)) * 100
                            value = self.callback(task_id, self.status, times)
                        else:
                            self.status = 'completed!'
                            time.sleep(180)
                            times = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                            value = self.callback(task_id, self.status, times)
                        if value:
                            self.p_rm_model(task_name, data_name)
                            break
                dest_ssh.close()
                dest_scp.close()
                dest_sftp.close()
            except Exception as e:
                current_app.logger.error(e)

    # 并行结束
    def p_rm_model(self, task_name, data_name):
        dress = parallel_training[0]
        dest_ssh = create_ssh_client(dress["dest_scp_ip"], dress["dest_scp_port"], dress["dest_scp_user"],
                                     dress["dest_scp_passwd"])
        dest_scp = scp.SCPClient(dest_ssh.get_transport())
        dest_sftp = dest_ssh.open_sftp()

        # path = "/data/deeplearning/train_platform/train_task/{}/output/models".format(task_name)
        path = os.path.join(data_sour_train, task_name, "output/models")
        files = dest_sftp.listdir(path)
        for i in files:
            if i.endswith("json"):
                dest_sftp.get(os.path.join(path, i), os.path.join(path, i))
                files.remove(i)
        files.sort()
        model_path = os.path.join(path, files[-1])
        dest_sftp.get(model_path, model_path)
        dest_ssh.close()
        dest_scp.close()
        dest_sftp.close()
        rm_datas(data_name)
        rm_folder(task_name)
        dest_scp, dest_sftp, dest_ssh = client(id="host")
        task_lists = os.listdir(data0_sour_train)
        if task_name not in task_lists:
            # dest_sftp.mkdir(os.path.join("/data/deeplearning/train_platform/train_task/", task_name))
            # copyFiles(os.path.join(data_sour_train, task_name), os.path.join(data0_sour_train, task_name))
            dest_scp.put(os.path.join(data_sour_train, task_name),
                         os.path.join(data0_sour_train, task_name),
                         recursive=True)

        dest_ssh.close()
        dest_scp.close()
        dest_sftp.close()
