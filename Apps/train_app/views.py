# -*-coding:utf-8-*-
from django.views import View
from django.db.models import Q
import json
import time
import zipfile
from subprocess import Popen, PIPE
import shutil
from utils.response_code import RET
from django.http import JsonResponse, FileResponse
from django.shortcuts import render
from rest_framework.views import APIView
from utils.util import recog_map, get_global_conf, make_folder, op_conf, TMP_PATH, TRAIN_CONF_PATH, TRAIN_RELEASE_PATH, WEIGHT_PATH, TRAIN_DATA, TRAIN_PATH, TRAIN_LOG_PATH, init_weight, TRAIN_MODEL_PATH, TRAIN_OUTPUT_PATH, writeAllFileToZip, train_pro_data
import os
from utils.scp_server import client
from train_platform import celery_app
from train_platform.settings import DOCKER_URL
from Apps.user_app.utils import get_username
from Apps.user_app.models import UserInfo
from Apps.user_app.serializer import TokenAuth
from .tasks import run_train
from .models import TrainTask
from Apps.data_app.models import DataProHandle
import logging

logger = logging.getLogger("django")

# Create your views here.


# 查询任务
class trainQuery(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        try:
            task_indexs = TrainTask.objects.all().order_by("-id")
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "database error!!"})
        else:
            data = []
            for task in list(task_indexs):
                data.append(task.to_dict())
            resp = JsonResponse({"errno": RET.OK, "data": data, "message": "success"})
        return resp

    def post(self, request):
        pass


# 建立任务
class trainBuild(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        old_id = request.GET.get('task_id')
        task_type = request.GET.get('task_type')
        init_w = os.path.join(WEIGHT_PATH, task_type)
        li = []
        if old_id:
            obj = TrainTask.objects.get(task_id=old_id)
            old_task_name = obj.task_name

            try:
                dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
                w_li = dest_sftp.listdir(TRAIN_RELEASE_PATH.format(old_task_name))
                dest_sftp.close()
                dest_scp.close()
                dest_ssh.close()
                li.extend(w_li)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())

        try:
            dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
            w_li = dest_sftp.listdir(init_w)
            dest_sftp.close()
            dest_scp.close()
            dest_ssh.close()
            li.extend(w_li)
            resp = JsonResponse({"errno": RET.OK, "data": li, "message": "success"})
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())

            resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "failed"})

        return resp

    def post(self, request):
        task_id = "TrainTask_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        type_id = req_dict["task_type"]
        old_id = req_dict["task_id"]
        version = req_dict["docker_tag"]
        data_name = req_dict["data_name"]
        comment = req_dict["comment"]
        task_name = req_dict["task_name"]
        gpu = req_dict["gpu"]
        weight = req_dict["weight"]
        task_type = str(type_id)
        host = "165"
        if old_id:
            obj = TrainTask.objects.get(task_id=old_id)
            old_task_name = obj.task_name
        else:
            old_task_name = ""
        if TrainTask.objects.filter(Q(task_type=task_type) & Q(task_name=task_name)).count() > 0:
            resp = JsonResponse({"errno": RET.DATAEXIST, "data": None, "message": "Task Existed"})
        else:
            make_folder(task_name, host)
            if weight:
                init_weight(host, old_task_name, task_name, weight, task_type)
            train_pro_data(host, data_name)
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="train", user_name=user_name.username)
            conf_count = len(data["conf"])
            try:
                obj = DataProHandle.objects.get(dir_name=data_name)
                train_task = TrainTask()

                train_task.task_id = task_id
                train_task.task_type = task_type
                train_task.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                train_task.host_id = host
                train_task.gpu_id = gpu
                train_task.task_name = task_name
                train_task.train_data_name = obj
                train_task.infos = comment
                train_task.version = version
                train_task.weight = weight

                train_task.save()
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "DB Error"})
            else:
                resp = JsonResponse({
                    "errno": RET.OK,
                    "data": {
                        "total": conf_count,
                        "task_id": task_id
                    },
                    "message": "success"
                })
        return resp


# 开始、继续、停止训练任务
class trainStart(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = TrainTask.objects.get(task_id=task_id)
        task_name = obj.task_name
        task_type = obj.task_type
        data_name = obj.train_data_name
        version = obj.version
        host_id = obj.host_id
        gpu_id = obj.gpu_id
        weight = obj.weight
        if not obj.status:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            input_name = os.path.join(TRAIN_DATA, data_name.dir_name)
            output_name = os.path.join(TRAIN_PATH, task_name)
            data = get_global_conf(task_type, version, types="train", user_name=user_name.username)
            cmd = data["cmd"]
            docker = DOCKER_URL + "{}:{}".format(task_type, version)
            weight_path = ""
            if weight:
                weight_path = os.path.join(TRAIN_MODEL_PATH.format(task_name), weight)
            run_task = run_train.apply_async(
                args=[task_name, docker, host_id, gpu_id, weight_path, task_id, input_name, output_name, cmd],
                countdown=5)
            # run_train(
            #     task_name, docker, host_id, gpu_id, weight_path, task_id, input_name, output_name, cmd)
            obj.status = "progress"
            obj.pro_id = run_task.id
            obj.save()

            u_obj = UserInfo.objects.get(username=user_name)
            if u_obj.default_docker:
                d = json.loads(u_obj.default_docker)
                if task_type in d:
                    if d[task_type] != version:
                        d[task_type] = version
                else:
                    d[task_type] = version
            else:
                d = {}
                d[task_type] = version
            u_obj.default_docker = json.dumps(d)
            u_obj.save()
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
        else:
            resp = JsonResponse({"errno": RET.DATAEXIST, "data": None, "message": "The task is already start"})
        return resp

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = req_dict["task_id"]
        op_type = req_dict["type"]
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        try:
            obj = TrainTask.objects.get(task_id=task_id)
            pro_id = obj.pro_id
            task_name = obj.task_name
            task_type = obj.task_type
            data_name = obj.train_data_name
            version = obj.version
            host_id = obj.host_id
            gpu_id = obj.gpu_id
            weight = obj.weight
            status = obj.status

            if int(op_type) == 1:
                if status in ["progress"]:
                    task = run_train.AsyncResult(pro_id)
                    if task:
                        if task.state == 'PENDING':
                            celery_app.control.revoke(pro_id, terminate=True)
                            dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
                            cmd_1 = "docker ps -a | grep {}".format(task_id)
                            stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
                            reslut = stdout1.read()
                            if reslut:
                                cmd_2 = "docker stop {}".format(task_id)
                                stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)

                            dest_scp.close()
                            dest_sftp.close()
                            dest_ssh.close()
                            obj.status = "stop"

                            obj.save()

                            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "task stop"})
                    else:
                        resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "task no exist or success"})
                else:
                    resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "task no exist or success"})
            elif int(op_type) == 2:
                if status in ["stop", "failed"]:
                    model_out = TRAIN_OUTPUT_PATH.format(task_name)
                    input_name = os.path.join(TRAIN_DATA, data_name.dir_name)
                    output_name = os.path.join(TRAIN_PATH, task_name)
                    data = get_global_conf(task_type, version, types="train", user_name=user_name.username)
                    cmd = data["cmd"]
                    docker = DOCKER_URL + "{}:{}".format(task_type, version)
                    dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
                    li = dest_sftp.listdir(model_out)

                    if len(li) >= 2:
                        weight_path = model_out
                    else:
                        weight_path = os.path.join(TRAIN_MODEL_PATH.format(task_name), weight)
                    cmd_1 = "docker ps -a | grep {}".format(task_id)
                    stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
                    if stdout1.read():
                        cmd_2 = "docker rm {}".format(task_id)
                        stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
                    run_task = run_train.apply_async(
                        args=[task_name, docker, host_id, gpu_id, weight_path, task_id, input_name, output_name, cmd],
                        countdown=5)
                    obj.status = "progress"
                    obj.pro_id = run_task.id
                    obj.save()
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                    resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
                else:
                    resp = JsonResponse({
                        "errno": RET.DATAEXIST,
                        "data": None,
                        "message": "Task in progress or completed"
                    })
            else:
                resp = JsonResponse({"errno": RET.PARAMERR, "data": None, "message": "params error"})
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "DB Error"})
        return resp


# 查询设置配置文件
class trainConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        page = int(request.GET.get('page'))
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        try:
            obj = TrainTask.objects.get(task_id=task_id)
            task_type = obj.task_type
            version = obj.version
            host = obj.host_id
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            data = get_global_conf(task_type, version, types="train", host=host, user_name=user_name.username)
            if page <= len(data["conf"]):
                conf_path = data["conf"][page - 1]
                conf_name = os.path.basename(conf_path)
                dest_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id))
                src_path = os.path.join(TMP_PATH, task_id)
                conf_data = op_conf(src_path=src_path,
                                    docker_type=task_type,
                                    docker_version=version,
                                    conf_path=conf_path,
                                    types=1,
                                    dest_path=dest_path)

                resp = JsonResponse({
                    "errno": RET.OK,
                    "data": {
                        "data": conf_data,
                        "conf_name": conf_name
                    },
                    "message": "success"
                })
            else:
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "Failed"})
        return resp

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = req_dict["task_id"]
        conf_name = req_dict["conf_name"]
        conf_json = req_dict["conf_data"]

        try:
            obj = TrainTask.objects.get(task_id=task_id)
            host = obj.host_id
            task_name = obj.task_name
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            src_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id))
            op_conf(src_path=src_path,
                    host=host,
                    types=2,
                    conf_name=conf_name,
                    conf_json=conf_json,
                    dest_path=TRAIN_CONF_PATH.format(task_name))
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
        return resp


# 上传下载配置文件
class opConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        pass

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = req_dict["task_id"]
        op_type = req_dict["type"]
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        # download
        if int(op_type) == 1:
            obj = TrainTask.objects.get(task_id=task_id)
            task_name = obj.task_name
            host_id = obj.host_id

            dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
            src = os.getcwd()
            if task_name in dest_sftp.listdir(TRAIN_PATH):
                conf_list = dest_sftp.listdir(TRAIN_CONF_PATH.format(task_name))
                dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name.username + task_id))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                for i in conf_list:
                    src_path = os.path.join(TRAIN_CONF_PATH.format(task_name), i)
                    dest_sftp.get(src_path, os.path.join(dest_path, i))
            else:
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
                dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
                if task_name in dest_sftp.listdir(TRAIN_PATH):
                    conf_list = dest_sftp.listdir(TRAIN_CONF_PATH.format(task_name))
                    dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name + task_id))
                    if not os.path.exists(dest_path):
                        os.makedirs(dest_path)
                    for i in conf_list:
                        src_path = os.path.join(TRAIN_CONF_PATH.format(task_name), i)
                        dest_sftp.get(src_path, os.path.join(dest_path, i))
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                else:
                    return JsonResponse({"errno": RET.NODATA, "data": None, "message": "Task No Exist"})
            zipFilePath = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_name + "_conf.zip"))
            zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
            os.chdir(dest_path)
            writeAllFileToZip(dest_path, zipFile)
            zipFile.close()
            shutil.rmtree(dest_path)
            os.chdir(src)
            src_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp'))
            name = task_name + "_conf.zip"
            files = open(os.path.join(src_path, name), 'rb')
            response = FileResponse(files)
            response['Content-Type'] = 'application/zip'
            response['Content-Disposition'] = 'attachment;filename=' + name

            os.remove(zipFilePath)
            return response
        else:
            pass


# 日志
class trainLog(APIView):

    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = TrainTask.objects.get(task_id=task_id)
        task_name = obj.task_name
        host_id = obj.host_id
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)

        dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
        src = os.getcwd()
        if task_name in dest_sftp.listdir(TRAIN_PATH):
            conf_list = dest_sftp.listdir(TRAIN_LOG_PATH.format(task_name))
            dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name.username + task_id))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            for i in conf_list:
                src_path = os.path.join(TRAIN_LOG_PATH.format(task_name), i)
                dest_sftp.get(src_path, os.path.join(dest_path, i))
        else:
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
            if task_name in dest_sftp.listdir(TRAIN_PATH):
                conf_list = dest_sftp.listdir(TRAIN_LOG_PATH.format(task_name))
                dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name.username + task_id))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                for i in conf_list:
                    src_path = os.path.join(TRAIN_LOG_PATH.format(task_name), i)
                    dest_sftp.get(src_path, os.path.join(dest_path, i))
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
            else:
                return JsonResponse({"errno": RET.NODATA, "data": None, "message": "Task No Exist"})
        zipFilePath = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_name + "_log.zip"))
        zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
        os.chdir(dest_path)
        writeAllFileToZip(dest_path, zipFile)
        zipFile.close()
        shutil.rmtree(dest_path)
        os.chdir(src)
        src_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp'))
        name = task_name + "_log.zip"
        files = open(os.path.join(src_path, name), 'rb')
        response = FileResponse(files)
        response['Content-Type'] = 'application/zip'
        response['Content-Disposition'] = 'attachment;filename=' + name
        os.remove(zipFilePath)
        return response

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = req_dict["task_id"]
        cmd = req_dict["cmd"]

        obj = TrainTask.objects.get(task_id=task_id)
        task_name = obj.task_name
        host_id = obj.host_id
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        log_name = ""

        dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
        if task_name in dest_sftp.listdir(TRAIN_PATH):
            conf_list = dest_sftp.listdir(TRAIN_LOG_PATH.format(task_name))
            dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name.username + task_id))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            for i in conf_list:
                log_name = i
                src_path = os.path.join(TRAIN_LOG_PATH.format(task_name), i)
                dest_sftp.get(src_path, os.path.join(dest_path, i))
        else:
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
            if task_name in dest_sftp.listdir(TRAIN_PATH):
                conf_list = dest_sftp.listdir(TRAIN_LOG_PATH.format(task_name))
                dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", user_name.username + task_id))
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
                for i in conf_list:
                    src_path = os.path.join(TRAIN_LOG_PATH.format(task_name), i)
                    log_name = i
                    dest_sftp.get(src_path, os.path.join(dest_path, i))
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
            else:
                return JsonResponse({"errno": RET.NODATA, "data": None, "message": "Task No Exist"})
        if cmd:
            result_cmd = cmd.format(os.path.join(dest_path, log_name))
        else:
            result_cmd = "tail -100 {}".format(os.path.join(dest_path, log_name))

        p = Popen(result_cmd, shell=True, stdout=PIPE)
        output = p.stdout.read().decode('UTF-8')
        lines = output.split(os.linesep)
        strs = "\n".join(lines)
        shutil.rmtree(dest_path)
        return JsonResponse({"errno": RET.OK, "data": strs, "message": "success"})


class Index(View):

    def get(self, request):

        return render(request, "indexs.html")
