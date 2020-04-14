# -*-coding:utf-8-*-
from django.db.models import Q
from utils.response_code import RET
from utils.scp_server import client
from utils.util import recog_map, get_global_conf, SRC_DATA, TRAIN_DATA, op_tar, download, op_conf, TMP_PATH, get_harbor_tag
from django.http import JsonResponse
from train_platform.settings import DOCKER_URL
from Apps.data_app.models import ProData, DataProHandle
from Apps.user_app.models import UserInfo
from rest_framework.views import APIView
from Apps.user_app.serializer import TokenAuth
from Apps.user_app.utils import get_username
from Apps.data_app.tasks import pro_data
import os
import json
import shutil
import logging
import time
logger = logging.getLogger("django")


# 本地数据上传
class localUpload(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        pass

    def post(self, request):
        # req_dict = json.loads(request.read())
        task_id = "ProData_" + str(int(time.time()))
        dirs = request.FILES
        dirlist = dirs.getlist('files')
        pathlist = request.POST.getlist('paths')
        task_type = "cls"
        info = "cls test"
        # host_id = request.POST.get("hostId")
        if not dirlist:
            resp = JsonResponse({"errno": RET.NODATA, "data": "", "message": "no found file"})
        else:
            total = len(dirlist)
            fail = 0
            succ = 0

            for file in dirlist:
                dir_name = pathlist[dirlist.index(file)].split('/')[:-1][0]
                position = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'upload_files')),
                                        '/'.join(pathlist[dirlist.index(file)].split('/')[:-1]))
                if not os.path.exists(position):
                    os.makedirs(position)
                try:
                    storage = open(position + '/' + file.name, 'wb+')
                    for chunk in file.chunks():
                        storage.write(chunk)
                    storage.close()
                except Exception as e:
                    fail += 1
                else:
                    succ += 1
            if (succ + fail) == total:
                state = "completed"
                dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
                src_path = os.path.abspath(os.path.join(os.getcwd(), 'upload_files', dir_name))
                if dir_name in dest_sftp.listdir(path=os.path.join(SRC_DATA, task_type)):
                    pass
                else:
                    dest_scp.put(src_path, os.path.join(SRC_DATA, task_type), recursive=True)
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                shutil.rmtree(os.path.abspath(os.path.join(os.getcwd(), 'upload_files', dir_name)))

                dir_name = pathlist[0].split('/')[:-1][0]
                pro_data = ProData()
                pro_data.task_id = task_id
                pro_data.pro_data_type = task_type
                pro_data.infos = info
                pro_data.status = state
                pro_data.pro_dir_name = dir_name
                pro_data.save()
                resp = JsonResponse({"errno": RET.OK, "data": {"task_id": task_id}, "message": "success"})
            else:
                resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "DB Error"})

        return resp


# 源数据
class srcData(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        type_id = request.GET.get("type")
        if str(type_id) == "1":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = {}
                for k, v in recog_map.items():
                    name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, v))
                    for name in name_list:
                        relust_list[name] = {"data_type": v, "dir_name": name, "infos": ""}

                data_list = ProData.objects.all()
                for data in list(data_list):
                    di = data.to_dict()
                    if di["dir_name"] not in relust_list:
                        relust_list[di["dir_name"]] = {
                            "data_type": di["data_type"],
                            "dir_name": di["dir_name"],
                            "infos": di["infos"]
                        }
                    else:
                        relust_list[di["dir_name"]]["infos"] = di["infos"]
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "DB Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        elif str(type_id) == "2":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = []
                for k, v in recog_map.items():
                    name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, v))
                    for name in name_list:
                        relust_list.append(name)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "IO Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        elif str(type_id) == "cls":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = []

                name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, str(type_id)))
                for name in name_list:
                    relust_list.append(name)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "IO Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        elif str(type_id) == "det":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = []

                name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, str(type_id)))
                for name in name_list:
                    relust_list.append(name)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "IO Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        elif str(type_id) == "seg":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = []

                name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, str(type_id)))
                for name in name_list:
                    relust_list.append(name)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "IO Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        elif str(type_id) == "depth":
            try:
                dest_scp, dest_sftp, dest_ssh = client("241", types="docker")
                relust_list = []

                name_list = dest_sftp.listdir(path=os.path.join(SRC_DATA, str(type_id)))
                for name in name_list:
                    relust_list.append(name)
            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "IO Error"})
            else:
                resp = JsonResponse({"errno": RET.OK, "data": relust_list, "message": "success"})
        else:
            resp = JsonResponse({"errno": RET.PARAMERR, "data": None, "message": "params error"})

        return resp

    def post(self, request):
        pass


# 标注数据预处理
class dataProHandle(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = DataProHandle.objects.get(task_id=task_id)
        task_type = obj.data_task_type
        data_name = obj.src_dir_name
        task_name = obj.dir_name
        version = obj.data_version
        host = obj.data_host_id
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        data = get_global_conf(task_type, version, types="prodata", user_name=user_name.username)
        cmd = data["cmd"]
        conf_path = os.path.join(TMP_PATH, task_id)
        input_path = os.path.join(SRC_DATA, task_type, data_name)
        output_path = os.path.join(TRAIN_DATA, task_name)
        docker = DOCKER_URL + "{}:{}".format(task_type, version)
        d = op_tar("241", os.path.join(SRC_DATA, task_type), data_name)
        if not d:
            rest = download(host, os.path.join(SRC_DATA, task_type, data_name + ".tar.gz"),
                            os.path.join(SRC_DATA, task_type))
            if not rest:
                dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
                cmd_1 = "rm -r {}".format(os.path.join(SRC_DATA, task_type, data_name + ".tar.gz"))
                stdin, stdout, stderr = dest_ssh.exec_command(cmd_1)
                print(stderr.read())
                s = pro_data.apply_async(args=[host, docker, cmd, input_path, output_path, conf_path, task_id],
                                         countdown=5)

                resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
            else:
                resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "Failed"})
        else:
            resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "Failed"})

        return resp

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = "DataProHandle_" + str(int(time.time()))
        type_id = req_dict["task_type"]
        old_id = req_dict["task_id"]
        version = req_dict["docker_tag"]
        data_name = req_dict["data_name"]
        comment = req_dict["comment"]
        task_name = req_dict["task_name"]
        task_type = str(type_id)
        # conf_json = req_dict["conf_json"]
        host = "165"

        if DataProHandle.objects.filter(Q(data_task_type=task_type) & Q(dir_name=task_name)).count() > 0:
            resp = JsonResponse({"errno": RET.DATAEXIST, "data": None, "message": "Data Existed"})
        else:
            try:
                obj = DataProHandle()
                obj.task_id = task_id
                obj.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                obj.dir_name = task_name
                obj.data_task_type = task_type
                obj.data_version = version
                obj.src_dir_name = data_name
                obj.data_host_id = host
                obj.infos = comment
                obj.status = "progress"
                obj.save()
                token = {"token": None}
                token["token"] = request.META.get('HTTP_TOKEN')
                user_name = get_username(token)
                data = get_global_conf(task_type, version, types="prodata", host=host, user_name=user_name.username)
                conf_count = len(data["conf"])

            except Exception as e:
                import traceback
                logger.error(traceback.format_exc())
                resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
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


# 识别类别
class recogClass(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        resp = JsonResponse({"errno": RET.OK, "data": recog_map, "message": "success"})
        return resp

    def post(self, request):
        pass


# docker版本查询
class DockerVer(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        types = request.GET.get("type")
        docker_type = str(types)
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        u_obj = UserInfo.objects.filter(username=user_name.username)
        if u_obj[0].default_docker:
            d = json.loads(u_obj[0].default_docker)
            if docker_type in d:
                default_docker = d[docker_type]
            else:
                default_docker = ""
        else:
            default_docker = ""
        result = get_harbor_tag(recog_type=types, types="kd-recog")
        resp = JsonResponse({
            "errno": RET.OK,
            "data": {
                "data": result,
                "default": default_docker
            },
            "message": "success"
        })
        return resp

    def post(self, request):
        pass


# 识别配置
class recogConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        type_id = request.GET.get('type')
        version = request.GET.get('docker')

        try:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            docker_type = str(type_id)
            data = get_global_conf(docker_type, version, user_name=user_name.username)
            if data:

                resp = JsonResponse({"errno": RET.OK, "data": data, "message": "success"})
            else:
                resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "No Data"})
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.IOERR, "data": None, "message": "Data IO Error"})

        return resp

    def post(self, request):
        pass


# 数据查询
class dataQuery(APIView):

    authentication_classes = (TokenAuth,)

    def get(self, request):
        try:
            all_data = DataProHandle.objects.all().order_by("-id")
            result = []
            for data in list(all_data):
                di = data.to_dict()
                result.append(di)
            resp = JsonResponse({"errno": RET.OK, "data": result, "message": "success"})
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})

        return resp

    def post(self, request):
        try:
            req_dict = json.loads(request.read())
            task_type = req_dict["task_type"]
            all_data = DataProHandle.objects.filter(data_task_type=task_type).order_by("-id")
            result = []
            for data in list(all_data):
                di = data.to_dict()
                result.append(di["dir_name"])
            resp = JsonResponse({"errno": RET.OK, "data": result, "message": "success"})
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        return resp


# 数据预处理配置
class dataConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        page = int(request.GET.get('page'))
        try:
            obj = DataProHandle.objects.get(task_id=task_id)
            task_type = obj.data_task_type
            version = obj.data_version
            host = obj.data_host_id
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="prodata", host=host, user_name=user_name.username)
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
            obj = DataProHandle.objects.get(task_id=task_id)
            host = obj.data_host_id
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
                    dest_path=os.path.join(TMP_PATH, task_id))
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
        return resp
