# -*-coding:utf-8-*-
import os
from train_platform.celery import app
from train_platform.settings import DOCKER_URL
from billiard.exceptions import Terminated
from utils.util import DOCKER_URL, client, EVAL_MODEL_PATH, TRAIN_RELEASE_PATH, EVAL_PATH, TRAIN_OUTPUT_PATH, model_pro_model, TEN_PATH, TRAIN_PATH
from Apps.train_app.models import TrainTask
from utils.util import SEG_DOCKER_URL
from .models import InferTestTask
import logging
import json
import time
logger = logging.getLogger("django")


@app.task(bind=True, throws=(Terminated,))
def model_eval(self, host, version, task_type, gpu, src_path, cmd, task_id, conf_path, types, use_model, conf_map):
    # def model_eval(host, version, task_type, gpu, src_path, cmd, task_id, conf_path, types, use_model, conf_map):
    total = 0
    success = 0
    for k, v in json.loads(use_model).items():
        total += len(v)
        model_obj = TrainTask.objects.get(task_name=k)
        host_id = model_obj.host_id
        dest_model = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k)
        if model_obj.status == "completed":
            for i in v:
                src = os.path.join(TRAIN_RELEASE_PATH.format(k), i)
                model_pro_model("241", src, host, dest_model, task_type, conf_map, k)
                weight = os.path.join(dest_model, i)
                out_path = os.path.join(EVAL_PATH, task_type, task_id, k, i)
                types = 1
                docker = DOCKER_URL + "{}:{}".format(task_type, version)
                run_cmd = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning --name {} {} {}".format(
                    task_id, docker, cmd.format(src_path, out_path, weight, gpu, conf_path, types))
                try:
                    dest_scp, dest_sftp, dest_ssh = client(ids=host)
                    stdin, stdout, stderr = dest_ssh.exec_command(run_cmd)

                    if stdout.readlines()[-1].strip() == "0":
                        success += 1
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                except Exception as e:
                    import traceback
                    logger.error(traceback.format_exc())

        else:
            for i in v:
                src = os.path.join(TRAIN_OUTPUT_PATH.format(k), i)
                model_pro_model(host_id, src, host, dest_model, task_type, conf_map, k)
                out_path = os.path.join(EVAL_PATH, task_type, task_id, k, i)
                weight = os.path.join(dest_model, i)
                docker = DOCKER_URL + "{}:{}".format(task_type, version)
                run_cmd = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning --name {} {} {}".format(
                    task_id, docker, cmd.format(src_path, out_path, weight, gpu, conf_path, types))
                print(run_cmd)
                try:
                    dest_scp, dest_sftp, dest_ssh = client(ids=host)
                    stdin, stdout, stderr = dest_ssh.exec_command(run_cmd)

                    if stdout.readlines()[-1].strip() == "0":
                        success += 1
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                except Exception as e:
                    import traceback
                    logger.error(traceback.format_exc())
    try:
        obj = InferTestTask.objects.get(task_id=task_id)
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        if total == success:
            obj.status = "success"
        else:
            obj.status = "failed"
        obj.save()
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())


@app.task(bind=True, throws=(Terminated,))
def model_con_test(self, host, version, task_type, gpu_id, src_path, cmd, task_id, conf_path, use_model, conf_map,
                   seg_tag, full_size):
    out_path_seg = os.path.join(EVAL_PATH, task_type, task_id, "recog_gt")
    out_path = os.path.join(EVAL_PATH, task_type, task_id)
    dest_model_seg = ""
    dest_model = ""
    for k, v in json.loads(use_model).items():
        model_obj = TrainTask.objects.get(task_name=k)
        host_id = model_obj.host_id
        dest_model_seg = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k)

        if model_obj.status == "completed":
            for i in v:
                dest_model = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k, i)
                src = os.path.join(TRAIN_RELEASE_PATH.format(k), i)
                model_pro_model("241", src, host, dest_model, task_type, conf_map, k)
        else:
            for i in v:
                dest_model = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k, i)
                src = os.path.join(TRAIN_OUTPUT_PATH.format(k), i)
                model_pro_model(host_id, src, host, dest_model, task_type, conf_map, k)
    docker_seg = SEG_DOCKER_URL + seg_tag
    cmd_1 = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning --name {} {} python /opt/t_segnet_server/consistency.py --model_type={} --model_dir={} --full_size={} --input_path={} --output_path={} --gpu_index={}".format(
        task_id, docker_seg, "", dest_model_seg, full_size, src_path, out_path_seg, gpu_id)
    obj = InferTestTask.objects.get(task_id=task_id)
    dest_scp, dest_sftp, dest_ssh = client(ids=host)
    try:
        dest_sftp.stat(out_path_seg)
    except IOError:
        cmd_3 = "mkdir -p {}".format(out_path_seg)
        stdin3, stdout3, stderr3 = dest_ssh.exec_command(cmd_3)
        print(stderr3.read())
    stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)

    if stdout1.readlines()[-1].strip() == "0":
        docker = DOCKER_URL + "{}:{}".format(task_type, version)
        cmd_2 = "nvidia-docker run -it --rm -v /data/deeplearning:/data/deeplearning {} {}".format(
            docker, cmd.format(src_path, out_path, dest_model, gpu_id, conf_path))
        stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
        if stdout2.readlines()[-1].strip() == "0":

            obj.status = "success"
            obj.result = "结果一致"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()

        else:
            obj.status = "failed"
            obj.result = "结果不一致"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()
    else:
        obj.status = "failed"
        obj.result = "结果不一致"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()


def model_con(host, version, task_type, gpu_id, src_path, cmd, task_id, conf_path, use_model, conf_map, seg_tag,
              full_size):
    out_path_seg = os.path.join(EVAL_PATH, task_type, task_id, "recog_gt")
    out_path = os.path.join(EVAL_PATH, task_type, task_id)
    dest_model_seg = ""
    dest_model = ""
    for k, v in json.loads(use_model).items():
        model_obj = TrainTask.objects.get(task_name=k)
        host_id = model_obj.host_id
        dest_model_seg = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k)

        if model_obj.status == "completed":
            for i in v:
                dest_model = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k, i)
                src = os.path.join(TRAIN_RELEASE_PATH.format(k), i)
                model_pro_model("241", src, host, dest_model_seg, task_type, conf_map, k)
        else:
            for i in v:
                dest_model = os.path.join(EVAL_MODEL_PATH, task_type, task_id, k, i)
                src = os.path.join(TRAIN_OUTPUT_PATH.format(k), i)
                model_pro_model(host_id, src, host, dest_model_seg, task_type, conf_map, k)
    docker_seg = SEG_DOCKER_URL + seg_tag
    cmd_1 = "nvidia-docker run -u $(id -u) --rm -v /data/deeplearning:/data/deeplearning --name {} {} python /opt/t_segnet_server/consistency.py --model_type={} --model_dir={} --full_size={} --input_path={} --output_path={} --gpu_index={}".format(
        task_id, docker_seg, "resnet-road", dest_model_seg, full_size, src_path, out_path_seg, gpu_id)
    print(cmd_1)
    obj = InferTestTask.objects.get(task_id=task_id)
    dest_scp, dest_sftp, dest_ssh = client(ids=host)
    try:
        dest_sftp.stat(out_path_seg)
    except IOError:
        cmd_3 = "mkdir -p {}".format(out_path_seg)
        stdin3, stdout3, stderr3 = dest_ssh.exec_command(cmd_3)
        print(stderr3.read())
    stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)

    if str(stdout1.readlines()[-1]).strip() == "0":
        docker = DOCKER_URL + "{}:{}".format(task_type, version)
        cmd_2 = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} {}".format(
            docker, cmd.format(src_path, out_path, dest_model, gpu_id, conf_path))
        print(cmd_2)
        stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)

        if str(stdout2.readlines()[-1]).strip() == "0":

            obj.status = "success"
            obj.result = "结果一致"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            return

        obj.status = "failed"
        obj.result = "结果不一致"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        return

    obj.status = "failed"
    obj.result = "结果不一致"
    obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    obj.save()
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    return


@app.task(bind=True, throws=(Terminated,))
def model_tensorRt(self, host_id, version, task_type, gpu_id, cmd, task_id, conf_path, use_model):
    weight = ""
    dest_path = os.path.join(TEN_PATH, task_type, task_id)
    for k, v in json.loads(use_model).items():
        src_path = os.path.join(TRAIN_RELEASE_PATH.format(k))
        weight = v

    dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
    try:
        dest_sftp.stat(dest_path)
    except IOError:
        cmd_1 = "mkdir -p {}".format(dest_path)

        stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
        print(stdout1.read())

    cmd_2 = "sshpass -p 12345678 scp -P 77 -r root@10.11.5.241:{}/* {}".format(src_path, dest_path)
    stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
    print(stdout2.read())

    docker = DOCKER_URL + "{}:{}".format(task_type, version)
    cmd_3 = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} {}".format(
        docker, cmd.format(dest_path, os.path.join(dest_path, weight[-1]), gpu_id, conf_path))
    print(cmd_3)
    obj = InferTestTask.objects.get(task_id=task_id)
    stdin3, stdout3, stderr3 = dest_ssh.exec_command(cmd_3)

    if str(stdout3.readlines()[-1]).strip() == "0":
        cmd_4 = "sshpass -p 12345678 scp -P 77 -r {}/* root@10.11.5.241:{}".format(dest_path, src_path)
        stdin4, stdout4, stderr24 = dest_ssh.exec_command(cmd_4)
        print(stdout4.read())
        obj.status = "success"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        return
    obj.status = "failed"
    obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    obj.save()
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    return


@app.task(bind=True, throws=(Terminated,))
def model_release(self, host_id, version, task_type, cmd, task_id, use_model):
    docker = DOCKER_URL + "{}:{}".format(task_type, version)
    task_name = ""
    weight = ""
    for k, v in json.loads(use_model).items():
        task_name = k
        weight = v[-1]
    cmd_1 = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning --name {} {} {}".format(
        task_id, docker, cmd.format(os.path.join(TRAIN_OUTPUT_PATH.format(task_name), weight)))

    dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
    stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
    obj = InferTestTask.objects.get(task_id=task_id)
    obj_train = TrainTask.objects.get(task_name=task_name)

    if stdout1.readlines()[-1].strip() == "0":
        cmd_2 = "sshpass -p 12345678 scp -P 77 -r {} root@10.11.5.241:{}".format(os.path.join(TRAIN_PATH, task_name),
                                                                                 TRAIN_PATH)
        stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
        print(stderr2.read())
        obj.status = "success"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        obj_train.status = "completed"
        obj_train.save()

        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        return
