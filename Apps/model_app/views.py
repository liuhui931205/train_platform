# -*-coding:utf-8-*-
import json
import time
import zipfile
import base64
import shutil
from utils.response_code import RET
from django.http import JsonResponse, FileResponse
from rest_framework.views import APIView
from utils.util import get_global_conf, op_conf, TMP_PATH, TEST_DATA, writeAllFileToZip, get_models, model_pro_data, EVAL_PATH, display_pro_data, copy_files, client, image_type_map, get_harbor_tag, model_manage
import os
from .tasks import model_eval, model_con, model_tensorRt, model_release
import requests
from Apps.user_app.utils import get_username
from Apps.user_app.models import UserInfo
from Apps.user_app.serializer import TokenAuth
from Apps.train_app.models import TrainTask
from .models import InferTestTask, ReleaseModel, ModelRecord
from multiprocessing import Manager, Process
import logging

logger = logging.getLogger("django")


class queueTask(object):

    def __init__(self, task_name="", host_id="", status=""):
        self.task_name = task_name
        self.host_id = host_id
        self.status = status


# 查询模型
class modelQuery(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_type = request.GET.get('task_type')
        manager = Manager()
        input_queue = manager.Queue()
        result_dict = manager.dict()
        try:
            task_indexs = TrainTask.objects.filter(task_type=task_type).order_by("-id")
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "database error!!"})
        else:
            datas = []
            for task in list(task_indexs):

                task_name = task.task_name
                host_id = task.host_id
                status = task.status
                queue_task = queueTask(task_name=task_name, host_id=host_id, status=status)
                input_queue.put(queue_task)
                d = {"name": task_name, "model": None, "info": task.infos}
                datas.append(d)
            pros = []
            if input_queue.qsize() > 5:

                for i in range(5):
                    p = Process(target=get_models, args=(input_queue, result_dict))
                    pros.append(p)
            else:
                p = Process(target=get_models, args=(input_queue, result_dict))
                pros.append(p)
            for i in pros:
                i.start()
            for i in pros:
                i.join()
            result = dict(result_dict)
            for i in datas:
                if i["name"] in result:
                    i["model"] = sorted(result[i["name"]], reverse=True)

            resp = JsonResponse({"errno": RET.OK, "data": datas, "message": "success"})
        return resp

    def post(self, request):
        pass


# 模型测试、验证
class modelTest(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        try:
            obj = InferTestTask.objects.get(task_id=task_id)
            task_type = obj.infer_task_type
            version = obj.infer_version
            use_model = obj.use_model
            data_name = obj.data_name
            types = obj.types
            gpu_id = obj.gpu_id
            host = obj.host_id
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)

        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            src_path = model_pro_data(host, task_id, json.loads(use_model), types, task_type, data_name)
            data = get_global_conf(task_type, version, types="test", user_name=user_name.username)
            cmd = data["cmd"]
            conf_map = data["conf_common"]
            # conf_map = ""
            conf_path = os.path.join(TMP_PATH, task_id)
            model_eval.apply_async(
                args=[host, version, task_type, gpu_id, src_path, cmd, task_id, conf_path, types, use_model, conf_map],
                countdown=5)
            # model_eval(host, version, task_type, gpu_id, src_path, cmd, task_id, conf_path, types, use_model, conf_map)

            resp = JsonResponse({"errno": RET.OK, "data": {"task_id": task_id}, "message": "success"})
        return resp

    def post(self, request):
        task_id = "InferTestTask_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        version = req_dict["docker_tag"]
        task_type = req_dict["task_type"]
        data_name = req_dict["data_name"]
        use_model = req_dict["models"]
        types = req_dict["types"]
        gpu = req_dict["gpu"]
        host_id = "165"
        try:
            obj = InferTestTask()
            obj.task_id = task_id
            obj.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.host_id = host_id
            obj.gpu_id = gpu
            obj.use_model = json.dumps(use_model)
            obj.infer_task_type = task_type
            obj.infer_version = version
            if int(types) == 2:
                data_name = "test_data"
            obj.data_name = data_name
            obj.types = types
            obj.status = "progress"
            obj.save()
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="test", host=host_id, user_name=user_name.username)
            conf_count = len(data["conf"])
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


# 模型测试、验证配置
class modelTestConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        page = int(request.GET.get('page'))
        try:
            obj = InferTestTask.objects.get(task_id=task_id)
            task_type = obj.infer_task_type
            version = obj.infer_version
            host = obj.host_id
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="test", host=host, user_name=user_name.username)
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
            obj = InferTestTask.objects.get(task_id=task_id)
            host = obj.host_id
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


# 可视化
class visualization(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = InferTestTask.objects.get(task_id=task_id)
        host = obj.host_id
        task_type = obj.infer_task_type
        types = obj.types
        use_model = obj.use_model
        status = obj.status
        if status == "success":
            src_path = os.path.join(TEST_DATA, task_type, task_id)
            dest_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id)))
            if not os.path.exists(dest_path):
                os.makedirs(dest_path)
            png_path = os.path.join(EVAL_PATH, task_type, task_id)
            display_pro_data(host, task_id, src_path, png_path, dest_path)
            total = 0
            for i in os.listdir(os.path.join(dest_path, "src")):
                if i.endswith("jpg"):
                    total += 1

            if str(types) == "1":
                resp = JsonResponse({
                    "errno": RET.OK,
                    "data": {
                        "task_id": task_id,
                        "result": None,
                        "total": total
                    },
                    "message": "success"
                })
            else:
                strs = ""
                dest_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id)))
                for k, v in json.loads(use_model).items():
                    for i in v:
                        with open(os.path.join(dest_path, k, i, "desc", "desc.txt"), 'r') as f:
                            strs += f.read() + "*" * 30 + "\n"

                resp = JsonResponse({
                    "errno": RET.OK,
                    "data": {
                        "task_id": task_id,
                        "result": strs,
                        "total": total
                    },
                    "message": "success"
                })
        else:
            resp = JsonResponse({"errno": RET.DATAERR, "data": None, "message": "failed"})
        return resp

    def post(self, request):
        req_dict = json.loads(request.read())
        task_id = req_dict["task_id"]
        cur_img = int(req_dict["cur_img"])
        obj = InferTestTask.objects.get(task_id=task_id)
        use_model = obj.use_model
        dest_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id)))
        if not os.path.exists(dest_path):
            resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "data not exist"})
        else:
            src_list = os.listdir(os.path.join(dest_path, "src"))
            img_list = []
            for i in src_list:
                if os.path.isdir(os.path.join(dest_path, "src", i)):
                    li = os.listdir(os.path.join(dest_path, "src", i))
                    for j in li:
                        img_list.append("{}/{}".format(i, j))
                else:
                    img_list = src_list

            img_list.sort()
            src_name = img_list[cur_img - 1]
            with open(os.path.join(dest_path, "src", src_name), "rb") as f:
                src = str(base64.b64encode(f.read()), encoding="utf-8")
            la_datas = []
            for k, v in json.loads(use_model).items():
                for i in v:
                    di = {}
                    di["name"] = k + "_" + i
                    label_name = src_name.split("/")[-1][:-4] + ".png"
                    with open(os.path.join(dest_path, k, i, "predict", label_name), 'rb') as f:
                        di["data"] = str(base64.b64encode(f.read()), encoding="utf-8")
                    la_datas.append(di)
            resp = JsonResponse({
                "errno": RET.OK,
                "data": {
                    "task_id": task_id,
                    "src_data": src,
                    "png_data": la_datas
                },
                "message": "success"
            })
        return resp


# 下载图片及指标
class downloadImage(APIView):

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
        obj = InferTestTask.objects.get(task_id=task_id)
        use_model = obj.use_model
        src = os.getcwd()
        if op_type == "1":
            zipFilePath = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, task_id + "_desc.zip"))
            zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
            for k, v in json.loads(use_model).items():
                for i in v:
                    dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, k, i, "desc"))
                    copy_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, k, i, k + i + "desc"))
                    copy_files(dest_path, copy_path)
                    os.chdir(copy_path)
                    writeAllFileToZip(copy_path, zipFile)
                    shutil.rmtree(copy_path)
            zipFile.close()
            # shutil.rmtree(dest_path)
            os.chdir(src)
            src_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id))
            name = task_id + "_desc.zip"
            files = open(os.path.join(src_path, name), 'rb')
            response = FileResponse(files)
            response['Content-Type'] = 'application/zip'
            response['Content-Disposition'] = 'attachment;filename=' + name

            os.remove(zipFilePath)
            return response
        else:
            zipFilePath = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, task_id + "_predict.zip"))
            zipFile = zipfile.ZipFile(zipFilePath, "w", zipfile.ZIP_DEFLATED)
            for k, v in json.loads(use_model).items():
                for i in v:
                    dest_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, k, i, "predict"))
                    copy_path = os.path.abspath(os.path.join(os.getcwd(), "tmp", task_id, k, i, k + i + "predict"))
                    copy_files(dest_path, copy_path)
                    os.chdir(copy_path)
                    writeAllFileToZip(copy_path, zipFile)
                    shutil.rmtree(copy_path)
            zipFile.close()
            # shutil.rmtree(dest_path)
            os.chdir(src)
            src_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp', task_id))
            name = task_id + "_predict.zip"
            files = open(os.path.join(src_path, name), 'rb')
            response = FileResponse(files)
            response['Content-Type'] = 'application/zip'
            response['Content-Disposition'] = 'attachment;filename=' + name

            os.remove(zipFilePath)
            return response


# 模型评估
class modelEval(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        pass

    def post(self, request):
        pass


# 模型一致性
class modelConsistency(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = InferTestTask.objects.get(task_id=task_id)
        host_id = obj.host_id
        version = obj.infer_version
        gpu_id = obj.gpu_id
        task_type = obj.infer_task_type
        use_model = obj.use_model
        seg_tag = obj.seg_tag
        image_type = obj.image_type
        full_size = image_type_map[image_type]

        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        src_path = os.path.join(os.path.abspath(os.path.join(os.getcwd(), 'demo', task_type)))

        dest_path = os.path.join(TEST_DATA, task_type, task_id)

        dest_scp, dest_sftp, dest_ssh = client(ids=host_id, types="docker")
        try:
            dest_sftp.stat(dest_path)
        except IOError:
            cmd = "mkdir -p {}".format(dest_path)
            stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd)
            print(stderr1.read())
        dest_scp.put(src_path, dest_path, recursive=True)
        cmd_2 = "cd {} && mv {}/* . && rm -r {}".format(dest_path, os.path.join(dest_path, task_type), task_type)
        stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
        print(stderr2.read())
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        data = get_global_conf(task_type, version, types="consistency", user_name=user_name.username)
        cmd = data["cmd"]
        conf_map = data["conf_common"]

        conf_path = os.path.join(TMP_PATH, task_id)
        # model_con.apply_async(
        #     args=[host_id, version, task_type, gpu_id, dest_path, cmd, task_id, conf_path,use_model, conf_map, seg_tag, full_size],
        #     countdown=5)
        model_con(host_id, version, task_type, gpu_id, dest_path, cmd, task_id, conf_path, use_model, conf_map,
                  seg_tag, full_size)
        resp = JsonResponse({"errno": RET.OK, "data": {"task_id": task_id}, "message": "success"})

        return resp

    def post(self, request):
        task_id = "Consistency_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        version = req_dict["docker_tag"]
        task_type = req_dict["task_type"]
        use_model = req_dict["models"]
        seg_tag = req_dict["seg_tag"]
        image_type = req_dict["image_type"]
        gpu = req_dict["gpu"]
        host_id = "165"
        if use_model:
            if len(use_model) > 1 or len(list(use_model.values())[0]) > 1:
                resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Only one model can be selected"})
            else:

                try:
                    obj = InferTestTask()
                    obj.task_id = task_id
                    obj.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                    obj.host_id = host_id
                    obj.gpu_id = gpu
                    obj.use_model = json.dumps(use_model)
                    obj.infer_task_type = task_type
                    obj.infer_version = version
                    obj.seg_tag = seg_tag
                    obj.image_type = image_type
                    obj.data_name = "consistency_data"
                    obj.types = "3"
                    obj.status = "progress"
                    obj.save()
                    token = {"token": None}
                    token["token"] = request.META.get('HTTP_TOKEN')
                    user_name = get_username(token)
                    data = get_global_conf(task_type,
                                           version,
                                           types="consistency",
                                           host=host_id,
                                           user_name=user_name.username)
                    conf_count = len(data["conf"])
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
        else:
            resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Please select the model first"})

        return resp


# 模型一致性配置
class modelConsistencyConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        page = int(request.GET.get('page'))
        try:
            obj = InferTestTask.objects.get(task_id=task_id)
            task_type = obj.infer_task_type
            version = obj.infer_version
            host = obj.host_id
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="consistency", host=host, user_name=user_name.username)
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
            obj = InferTestTask.objects.get(task_id=task_id)
            host = obj.host_id
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


# 模型转TensorRt
class modelToTensorRt(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = InferTestTask.objects.get(task_id=task_id)
        host_id = obj.host_id
        version = obj.infer_version
        gpu_id = obj.gpu_id
        task_type = obj.infer_task_type
        use_model = obj.use_model
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)

        data = get_global_conf(task_type, version, types="tensorRt", user_name=user_name.username)
        cmd = data["cmd"]
        conf_path = os.path.join(TMP_PATH, task_id)
        model_tensorRt.apply_async(args=[host_id, version, task_type, gpu_id, cmd, task_id, conf_path, use_model],
                                   countdown=5)
        # model_tensorRt(host_id, version, task_type, gpu_id, cmd, task_id, conf_path, use_model)
        resp = JsonResponse({"errno": RET.OK, "data": {"task_id": task_id}, "message": "success"})

        return resp

    def post(self, request):
        task_id = "TensorRt_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        version = req_dict["docker_tag"]
        task_type = req_dict["task_type"]
        use_model = req_dict["models"]
        gpu = req_dict["gpu"]
        host_id = "165"
        if use_model:
            if len(use_model) > 1 or len(list(use_model.values())[0]) > 1:
                resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Only one model can be selected"})
            else:
                for k, v in use_model.items():
                    obj_train = TrainTask.objects.get(task_name=k)
                if obj_train.status != "completed":
                    resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Please release first"})
                else:
                    try:
                        obj = InferTestTask()
                        obj.task_id = task_id
                        obj.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                        obj.host_id = host_id
                        obj.gpu_id = str(gpu)
                        obj.use_model = json.dumps(use_model)
                        obj.infer_task_type = task_type
                        obj.infer_version = version
                        obj.types = "4"
                        obj.status = "progress"
                        obj.save()
                        token = {"token": None}
                        token["token"] = request.META.get('HTTP_TOKEN')
                        user_name = get_username(token)
                        data = get_global_conf(task_type,
                                               version,
                                               types="tensorRt",
                                               host=host_id,
                                               user_name=user_name.username)
                        conf_count = len(data["conf"])
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

        else:

            resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Please select the model first"})
        return resp


# 模型转TensorRt配置
class modelTensorRtConf(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        page = int(request.GET.get('page'))
        try:
            obj = InferTestTask.objects.get(task_id=task_id)
            task_type = obj.infer_task_type
            version = obj.infer_version
            host = obj.host_id
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())
            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "Failed"})
        else:
            token = {"token": None}
            token["token"] = request.META.get('HTTP_TOKEN')
            user_name = get_username(token)
            data = get_global_conf(task_type, version, types="tensorRt", host=host, user_name=user_name.username)
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
            obj = InferTestTask.objects.get(task_id=task_id)
            host = obj.host_id
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


# 模型发布
class modelRelease(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        obj = InferTestTask.objects.get(task_id=task_id)
        host_id = obj.host_id
        version = obj.infer_version
        task_type = obj.infer_task_type
        use_model = obj.use_model
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)

        data = get_global_conf(task_type, version, types="release", user_name=user_name.username)
        cmd = data["cmd"]
        model_release.apply_async(args=[host_id, version, task_type, cmd, task_id, use_model], countdown=5)
        resp = JsonResponse({"errno": RET.OK, "data": {"task_id": task_id}, "message": "success"})

        return resp

    def post(self, request):
        task_id = "Release_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        version = req_dict["docker_tag"]
        task_type = req_dict["task_type"]
        use_model = req_dict["models"]
        if use_model:
            if len(use_model) > 1 or len(list(use_model.values())[0]) > 1:
                resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Only one model can be selected"})
            else:
                for k, v in use_model.items():
                    obj = TrainTask.objects.get(task_name=k)
                host_id = obj.host_id
                if obj.status == "completed":
                    resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "The model has been released"})
                else:
                    try:
                        obj = InferTestTask()
                        obj.task_id = task_id
                        obj.start_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                        obj.host_id = host_id
                        obj.use_model = json.dumps(use_model)
                        obj.infer_task_type = task_type
                        obj.infer_version = version
                        obj.types = "5"
                        obj.status = "progress"
                        obj.save()
                        token = {"token": None}
                        token["token"] = request.META.get('HTTP_TOKEN')
                        user_name = get_username(token)
                        data = get_global_conf(task_type,
                                               version,
                                               types="release",
                                               host=host_id,
                                               user_name=user_name.username)
                        conf_count = len(data["conf"])
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

        else:

            resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Please select the model first"})
        return resp


# 模型上线
class modelOnline(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        task_id = request.GET.get('task_id')
        pass

    def post(self, request):
        task_id = "Online_" + str(int(time.time()))
        req_dict = json.loads(request.read())
        types = req_dict["types"]
        sub_type = req_dict["sub_type"]
        scene = req_dict["scene"]
        adcode = req_dict["adcode"]
        biz = req_dict["biz"]
        customer = req_dict["customer"]
        use_model = req_dict["model"]
        file_path = req_dict["file_path"]
        description = req_dict["description"]
        name = req_dict["name"]
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        user_name = get_username(token)
        if use_model:
            if len(use_model) > 1 or len(list(use_model.values())[0]) > 1:
                resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Only one model can be selected"})
            else:
                try:

                    qur_obj = ReleaseModel.objects.filter(model=list(use_model.keys())[0])
                    if list(qur_obj):
                        if qur_obj[0].status == "success":
                            resp = JsonResponse({"errno": RET.DATAEXIST, "data": None, "message": "模型已发布"})
                        else:
                            params = {
                                "author": user_name.username,
                                "biz": str(biz),
                                "customer": customer,
                                "date": str(time.strftime("%Y-%m-%d %H:%M", time.localtime())),
                                "description": description,
                                "filePath": file_path,
                                "name": name,
                                "type": str(types),
                                "subType": str(sub_type),
                                "adcode": str(adcode),
                                "scene": str(scene)
                            }

                            url = model_manage
                            resp = requests.post(url, json=params)
                            if json.loads(resp.text)["code"] == "0":
                                qur_obj[0].status = "success"
                                qur_obj[0].save()
                                resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
                            else:
                                qur_obj[0].status = "failed"
                                qur_obj[0].save()
                                resp = JsonResponse({"errno": RET.THIRDERR, "data": None, "message": "failed"})

                    else:

                        params = {
                            "author": user_name.username,
                            "biz": str(biz),
                            "customer": customer,
                            "date": str(time.strftime("%Y-%m-%d %H:%M", time.localtime())),
                            "description": description,
                            "filePath": file_path,
                            "name": name,
                            "type": str(types),
                            "subType": str(sub_type),
                            "adcode": str(adcode),
                            "scene": str(scene)
                        }

                        url = model_manage
                        resp = requests.post(url, json=params)
                        if json.loads(resp.text)["code"] == "0":
                            obj = ReleaseModel()
                            obj.task_id = task_id
                            obj.release_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                            obj.types = types
                            obj.sub_type = sub_type
                            obj.scene = scene
                            obj.adcode = adcode
                            obj.biz = biz
                            obj.author = user_name.username
                            obj.file_path = file_path
                            obj.description = description
                            obj.model = list(use_model.keys())[0]
                            obj.customer = customer
                            obj.status = "success"
                            obj.save()
                            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
                        else:
                            obj = ReleaseModel()
                            obj.task_id = task_id
                            obj.release_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                            obj.types = types
                            obj.sub_type = sub_type
                            obj.scene = scene
                            obj.adcode = adcode
                            obj.biz = biz
                            obj.author = user_name.username
                            obj.file_path = file_path
                            obj.description = description
                            obj.model = list(use_model.keys())[0]
                            obj.customer = customer
                            obj.status = "failed"
                            obj.save()
                            resp = JsonResponse({"errno": RET.THIRDERR, "data": None, "message": "failed"})

                except Exception as e:
                    import traceback
                    logger.error(traceback.format_exc())
                    resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "DB Error"})
        else:
            resp = JsonResponse({"errno": RET.NODATA, "data": None, "message": "Please select the model first"})

        return resp


# 查询识别服务版本
class querySegnet(APIView):
    authentication_classes = (TokenAuth,)

    def get(self, request):
        data = get_harbor_tag(types="kd-mr")
        resp = JsonResponse({"errno": RET.OK, "data": data, "message": "success"})

        return resp

    def post(self, request):

        pass
