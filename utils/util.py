# -*-coding:utf-8-*-
from subprocess import Popen, PIPE
from utils.scp_server import client, server
from train_platform.settings import DOCKER_URL
import os
import json
import shutil
import time
import requests
from operator import itemgetter
from Apps.train_app.models import TrainTask
recog_map = {"1": "seg", "2": "det", "3": "cls", "4": "depth"}

task_map = {
    "type": {
        "1": "分割",
        "2": "检测",
        "3": "分类",
        "4": "深度",
        "5": "点云"
    },
    "subType": {
        "0": "通用",
        "1": "半图",
        "2": "全图"
    },
    "scene": {
        "0": "通用",
        "1": "高速",
        "2": "城市"
    },
    "biz": {
        "0": "其他",
        "1": "组网",
        "2": "病害",
        "3": "铁路异物"
    },
    "customer": {
        "0": "其他",
        "1": "宽凳",
        "2": "上汽",
        "3": "广汽"
    }
}

model_manage = "http://192.168.5.34:32800/model-management/models/set"

path_map = {"seg": "kd-seg", "cls": "classification", "det": "kd-det", "depth": "monodepth2"}

model_map = {"seg": "json", "cls": "json", "det": "", "depth": ""}

model_op_map = {"1": "测试", "2": "验证", "3": "一致性", "4": "TensorRt", "5": "发布"}

image_type_map = {"full": 1, "half": 2}

SEG_DOCKER_URL = "kd-bd02.kuandeng.com/kd-mr/lane:"

SRC_DATA = "/data/deeplearning/train_platform/src_data/train_data/"
TEST_DATA = "/data/deeplearning/train_platform/src_data/eval_data/"
TRAIN_DATA = "/data/deeplearning/train_platform/train_data/"
TRAIN_PATH = "/data/deeplearning/train_platform/train_task/"
TRAIN_LOG_PATH = "/data/deeplearning/train_platform/train_task/{}/log/"
TRAIN_OUTPUT_PATH = "/data/deeplearning/train_platform/train_task/{}/output/"
TRAIN_MODEL_PATH = "/data/deeplearning/train_platform/train_task/{}/model/"
TRAIN_RELEASE_PATH = "/data/deeplearning/train_platform/train_task/{}/release/"
TRAIN_CONF_PATH = "/data/deeplearning/train_platform/train_task/{}/conf/"
EVAL_PATH = "/data/deeplearning/train_platform/eval_task/"
EVAL_MODEL_PATH = "/data/deeplearning/train_platform/src_data/eval_model"
CONS_PATH = "/data/deeplearning/train_platform/consistency/"
TEN_PATH = "/data/deeplearning/train_platform/TensorRt/"
GLOBAL_CONF = "/opt/{}/config.json"
TMP_PATH = "/data/deeplearning/train_platform/tmp/"
WEIGHT_PATH = "/data/deeplearning/train_platform/weights/"


# 获取全局配置
def get_global_conf(docker_type, docker_version, types=None, host='165', user_name=""):
    global_conf = GLOBAL_CONF.format(path_map[docker_type])
    docker = DOCKER_URL + "{}:{}".format(docker_type, docker_version)
    cmd_2 = "sshpass -p kd-123 ssh kdreg@10.11.5.{} nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} cp {} {}".format(
        host, docker, global_conf, TMP_PATH)

    p = Popen(cmd_2, shell=True, stdout=PIPE)
    output = p.stdout.read().decode('UTF-8')
    if output:
        return 0
    dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
    dest_path = os.path.abspath(os.path.join(os.getcwd(), 'tmp', user_name))
    if not os.path.exists(dest_path):
        os.makedirs(dest_path)
    dest_sftp.get(os.path.join(TMP_PATH, "config.json"), os.path.join(dest_path, "config.json"))
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    with open(os.path.join(dest_path, "config.json"), 'r') as f:
        json_data = json.load(f)
    if types:

        if types in json_data:
            shutil.rmtree(dest_path)
            return json_data[types]
        else:
            shutil.rmtree(dest_path)
            return 0
    else:
        shutil.rmtree(dest_path)
        return json_data


# 操作功能配置文件
def op_conf(src_path="",
            docker_type="",
            docker_version="",
            conf_path="",
            types=1,
            host="165",
            conf_name="",
            conf_json="",
            dest_path=""):
    if int(types) == 1:
        dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
        tmp_path = os.path.dirname(src_path)
        name = os.path.basename(src_path)
        docker = DOCKER_URL + "{}:{}".format(docker_type, docker_version)
        if name not in dest_sftp.listdir(tmp_path):
            cmd = "mkdir -p {}".format(src_path)
            stdin, stdout, stderr = dest_ssh.exec_command(cmd)
            print(stdout.read())
        cmd_2 = "sshpass -p kd-123 ssh kdreg@10.11.5.{} nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} cp {} {}".format(
            host, docker, conf_path, src_path)

        p = Popen(cmd_2, shell=True, stdout=PIPE)
        output = p.stdout.read().decode('UTF-8')
        if output:
            return 0
        conf_name = os.path.basename(conf_path)
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        dest_sftp.get(os.path.join(src_path, conf_name), os.path.join(dest_path, conf_name))
        # cmd = "rm -r {}".format(src_path)
        # stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        if conf_name.endswith("json"):
            with open(os.path.join(dest_path, conf_name), 'r') as f:
                json_data = json.load(f)
        else:
            with open(os.path.join(dest_path, conf_name), 'r') as f:
                json_data = f.read()
        # shutil.rmtree(dest_path)
        return json_data
    else:

        if conf_name.endswith("json"):
            with open(os.path.join(src_path, conf_name), 'w') as f:
                f.write(json.dumps(conf_json))
        else:
            with open(os.path.join(src_path, conf_name), 'w') as f:
                f.write(conf_json)
        dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")

        dest_sftp.put(os.path.join(src_path, conf_name), os.path.join(dest_path, conf_name))
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        shutil.rmtree(src_path)
        return 1


# 打包操作
def op_tar(host, path, dir_name, type=1):
    if type == 1:
        tar_cmd = "cd {} && tar -czvf {}/{}.tar.gz {}".format(path, path, dir_name, dir_name)
    else:
        tar_cmd = "cd {} && tar -xzvf {}/{}.tar.gz && rm {}/{}.tar.gz".format(path, path, dir_name, path, dir_name)
    dest_scp, dest_sftp, dest_ssh = client(ids=str(host), types="docker")
    stdin, stdout, stderr = dest_ssh.exec_command(tar_cmd)
    # time.sleep(10)
    if not stderr.read():
        for i in stdout.readlines():
            continue
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        return 0
    return 1


# 上传
def upload(host, src_path, dest_path):
    path = os.path.dirname(src_path)
    name = os.path.basename(src_path)
    rest = op_tar(host, path, name)
    if not rest:
        tar_cmd = "sshpass -p 12345678 scp -P 77 -r {} root@10.11.5.241:{}".format(src_path, dest_path)
        dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
        stdin, stdout, stderr = dest_ssh.exec_command(tar_cmd)
        if not stderr.read():
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()

            return 0
    return 1


# 下载
def download(host, src_path, dest_path):
    # rest = op_tar(host, dest_path, name, type=1)

    tar_cmd = "sshpass -p 12345678 scp -P 77 -r root@10.11.5.241:{} {}".format(src_path, dest_path)
    dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
    stdin, stdout, stderr = dest_ssh.exec_command(tar_cmd)

    if not stderr.read():
        name = os.path.basename(src_path)[:-7]
        rest = op_tar(host, dest_path, name, type=2)

        if not rest:
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            return 0
    return 1


# 建立任务文件夹
def make_folder(task_name, host):
    dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
    if task_name not in dest_sftp.listdir(TRAIN_PATH):
        cmd = "mkdir -p {}".format(os.path.join(TRAIN_PATH, task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    if os.path.basename(TRAIN_LOG_PATH) not in dest_sftp.listdir(os.path.join(TRAIN_PATH, task_name)):
        cmd = "mkdir -p {}".format(TRAIN_LOG_PATH.format(task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    if os.path.basename(TRAIN_OUTPUT_PATH) not in dest_sftp.listdir(os.path.join(TRAIN_PATH, task_name)):
        cmd = "mkdir -p {}".format(TRAIN_OUTPUT_PATH.format(task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    if os.path.basename(TRAIN_MODEL_PATH) not in dest_sftp.listdir(os.path.join(TRAIN_PATH, task_name)):
        cmd = "mkdir -p {}".format(TRAIN_MODEL_PATH.format(task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    if os.path.basename(TRAIN_RELEASE_PATH) not in dest_sftp.listdir(os.path.join(TRAIN_PATH, task_name)):
        cmd = "mkdir -p {}".format(TRAIN_RELEASE_PATH.format(task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    if os.path.basename(TRAIN_CONF_PATH) not in dest_sftp.listdir(os.path.join(TRAIN_PATH, task_name)):
        cmd = "mkdir -p {}".format(TRAIN_CONF_PATH.format(task_name))
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        print(stdout.read())
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()


# 初始模型
def init_weight(host, old_task_name, task_name, weight, task_type):
    if old_task_name:
        dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
        if weight in dest_sftp.listdir(os.path.join(WEIGHT_PATH, task_type)):
            src_path = os.path.join(WEIGHT_PATH, task_type, weight)
        else:
            src_path = os.path.join(TRAIN_RELEASE_PATH.format(old_task_name), weight)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
    else:
        src_path = os.path.join(WEIGHT_PATH, task_type, weight)
    dest_path = os.path.join(TRAIN_MODEL_PATH.format(task_name), weight)
    tar_cmd = "sshpass -p 12345678 scp -P 77 -r root@10.11.5.241:{} {} ".format(src_path, dest_path)
    dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
    stdin, stdout, stderr = dest_ssh.exec_command(tar_cmd)
    print(stdout.read())
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()


# 压缩
def writeAllFileToZip(absDir, zipFile):

    for f in os.listdir(absDir):
        absFile = os.path.join(absDir, f)    #子文件的绝对路径
        if os.path.isdir(absFile):    #判断是文件夹，继续深度读取。
            relFile = absFile[len(os.getcwd()) + 1:]    #改成相对路径，否则解压zip是/User/xxx开头的文件。
            zipFile.write(relFile)    #在zip文件中创建文件夹
            writeAllFileToZip(absFile, zipFile)    #递归操作
        else:    #判断是普通文件，直接写到zip文件中。
            relFile = absFile[len(os.getcwd()) + 1:]    #改成相对路径
            zipFile.write(relFile)
    return


# 查询每个任务的模型
def get_models(input_queue, result_dict):
    while 1:
        if input_queue.empty():
            break
        task = input_queue.get()
        task_name = task.task_name
        host_id = task.host_id
        status = task.status
        if status == "completed":
            host_id = "241"
            dest_scp, dest_sftp, dest_ssh = client(ids=str(host_id), types="docker")
            li = dest_sftp.listdir(TRAIN_RELEASE_PATH.format(task_name))
            result_dict[task_name] = li
        else:
            dest_scp, dest_sftp, dest_ssh = client(ids=str(host_id))
            li = dest_sftp.listdir(TRAIN_OUTPUT_PATH.format(task_name))
            result_dict[task_name] = li

        dest_sftp.close()
        dest_scp.close()
        dest_ssh.close()


# 准备训练数据
def train_pro_data(host, input_name):
    dest_scp, dest_sftp, dest_ssh = client(ids=host)
    if input_name in dest_sftp.listdir(TRAIN_DATA):
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        return
    d = op_tar("241", TRAIN_DATA, input_name)
    if not d:
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        download(host, os.path.join(TRAIN_DATA, input_name + ".tar.gz"), TRAIN_DATA)
        cmd = "rm -r {}".format(os.path.join(TRAIN_DATA, input_name + ".tar.gz"))
        dest_scp, dest_sftp, dest_ssh = client(ids="241", types="docker")
        stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd)
        print(stdout1.read())
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()


# 准备测试验证数据
def model_pro_data(host, task_id, use_model, types, task_type, data_name):
    if types == "1":
        d = op_tar("241", os.path.join(TEST_DATA, task_type), data_name)
        if not d:
            rest = download(host, os.path.join(TEST_DATA, task_type, data_name + ".tar.gz"),
                            os.path.join(TEST_DATA, task_type))
            if not rest:
                return os.path.join(TEST_DATA, task_type, data_name)
        return 0

    else:
        dest_path = os.path.join(TEST_DATA, task_type, task_id)

        for k, v in use_model.items():
            obj = TrainTask.objects.get(task_name=k)
            train_data_name = obj.train_data_name
            src_path = os.path.join(TRAIN_DATA, train_data_name.dir_name, "val")
            tar_cmd = "cd {} && tar -czvf {}/{}.tar.gz *".format(src_path, src_path, task_id)
            dest_scp, dest_sftp, dest_ssh = client(ids=str(241), types="docker")
            stdin1, stdout1, stderr1 = dest_ssh.exec_command(tar_cmd)
            if not stderr1.read():
                for i in stdout1.readlines():
                    continue
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()

                cp_cmd = "sshpass -p 12345678 scp -P 77 -r root@10.11.5.241:{} {}".format(
                    os.path.join(src_path, task_id + ".tar.gz"), dest_path)
                dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
                if task_id not in dest_sftp.listdir(os.path.join(TEST_DATA, task_type)):
                    stdin2, stdout2, stderr2 = dest_ssh.exec_command("mkdir -p {}".format(dest_path))
                    print(stdout2.read())
                stdin3, stdout3, stderr3 = dest_ssh.exec_command(cp_cmd)
                print(stdout3.read())
                tr_cmd = "cd {} && tar -xzvf {}/{}.tar.gz && rm {}/{}.tar.gz".format(
                    dest_path, dest_path, task_id, dest_path, task_id)
                stdin4, stdout4, stderr4 = dest_ssh.exec_command(tr_cmd)
                if not stderr4.read():
                    for j in stdout4.readlines():
                        continue
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                cmd = "rm -r {}".format(os.path.join(src_path, task_id + ".tar.gz"))
                dest_scp, dest_sftp, dest_ssh = client(ids=str(241), types="docker")
                stdin5, stdout5, stderr5 = dest_ssh.exec_command(cmd)
                print(stdout5.read())
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()

            else:
                return 0
        return dest_path


# 准备测试验证模型
def model_pro_model(src_host, src_path, dest_host, dest_path, task_type, conf_map, task_name):
    if str(src_host) != str(dest_host):
        dest_scp, dest_sftp, dest_ssh = client(ids=str(dest_host), types="docker")
        try:
            dest_sftp.stat(dest_path)
        except IOError:
            cmd_1 = "mkdir -p {}".format(dest_path)
            stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
            r = stderr1.read()
            if r:
                print(stdout1.read())
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
                return
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        ext = model_map[task_type]
        dest_scp, dest_sftp, dest_ssh = client(ids=dest_host, types="docker")

        if ext:
            ext_path = os.path.dirname(src_path) + "/*.{}".format(ext)
            cmd_2 = "sshpass -p {} scp -P {} root@10.11.5.{}:{} {}".format(
                server[str(src_host)]["docker"]["dest_scp_passwd"], server[str(src_host)]["docker"]["dest_scp_port"],
                str(src_host), ext_path, dest_path)
            stdin2, stdout2, stderr2 = dest_ssh.exec_command(cmd_2)
            print(stdout2.read())
        cmd_3 = "sshpass -p {} scp -P {} root@10.11.5.{}:{} {}".format(
            server[str(src_host)]["docker"]["dest_scp_passwd"], server[str(src_host)]["docker"]["dest_scp_port"],
            str(src_host), src_path, dest_path)

        stdin3, stdout3, stderr3 = dest_ssh.exec_command(cmd_3)
        print(stdout3.read())
        if conf_map:

            for i in conf_map:
                cmd_4 = "sshpass -p {} scp -P {} root@10.11.5.{}:{} {}".format(
                    server[str(src_host)]["docker"]["dest_scp_passwd"],
                    server[str(src_host)]["docker"]["dest_scp_port"], str(src_host),
                    os.path.join(TRAIN_CONF_PATH.format(task_name), i), dest_path)

                stdin4, stdout4, stderr4 = dest_ssh.exec_command(cmd_4)
                print(stdout4.read())

        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
    else:
        dest_scp, dest_sftp, dest_ssh = client(ids=str(dest_host), types="docker")
        try:
            dest_sftp.stat(dest_path)
        except IOError:
            cmd_1 = "mkdir -p {}".format(dest_path)
            stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
            r = stderr1.read()
            if r:
                print(r)
                print(stdout1.read())
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
                return

        ext = model_map[task_type]
        if ext:
            ext_path = os.path.dirname(src_path) + "/*.{}".format(ext)
            cmd_4 = "cp {} {}".format(ext_path, dest_path)
            stdin4, stdout4, stderr4 = dest_ssh.exec_command(cmd_4)
            print(stdout4.read())
        cmd_5 = "cp {} {}".format(src_path, dest_path)
        stdin5, stdout5, stderr5 = dest_ssh.exec_command(cmd_5)
        print(stdout5.read())

        if conf_map:

            for i in conf_map:
                cmd_6 = "cp {} {}".format(os.path.join(TRAIN_CONF_PATH.format(task_name), i), dest_path)

                stdin6, stdout6, stderr6 = dest_ssh.exec_command(cmd_6)
                print(stdout6.read())

        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()


# 可视化准备源数据
def display_pro_data(host, task_id, src_path, png_path, dest_path):

    cmd_1 = "cd {} && tar -czvf {}/{}.tar.gz *".format(src_path, src_path, task_id)
    dest_scp, dest_sftp, dest_ssh = client(ids=str(host), types="docker")
    stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd_1)
    if not stderr1.read():
        for i in stdout1.readlines():
            continue

        dest_sftp.get(os.path.join(src_path, task_id + ".tar.gz"), os.path.join(dest_path, task_id + ".tar.gz"))

        cmd_2 = "mkdir -p {} && cd {} && tar -xzvf {}/{}.tar.gz && rm {}/{}.tar.gz".format(
            os.path.join(dest_path, "src"), os.path.join(dest_path, "src"), dest_path, task_id, dest_path, task_id)
        os.system(cmd_2)
        cmd_3 = "rm -r {}".format(os.path.join(src_path, task_id + ".tar.gz"))
        stdin3, stdout3, stderr3 = dest_ssh.exec_command(cmd_3)
        print(stderr3.read())

        cmd_4 = "cd {} && tar -czvf {}/{}.tar.gz *".format(png_path, png_path, task_id)
        stdin4, stdout4, stderr4 = dest_ssh.exec_command(cmd_4)
        print(stderr4.read())
        dest_sftp.get(os.path.join(png_path, task_id + ".tar.gz"), os.path.join(dest_path, task_id + ".tar.gz"))

        cmd_5 = "cd {} && tar -xzvf {}/{}.tar.gz && rm {}/{}.tar.gz".format(dest_path, dest_path, task_id, dest_path,
                                                                            task_id)
        os.system(cmd_5)

        cmd_6 = "rm -r {}".format(os.path.join(png_path, task_id + ".tar.gz"))
        stdin6, stdout6, stderr6 = dest_ssh.exec_command(cmd_6)
        print(stderr6.read())

        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()


# 复制
def copy_files(sourceDir, targetDir):
    for f in os.listdir(sourceDir):
        sourceF = os.path.join(sourceDir, f)
        targetF = os.path.join(targetDir, f)
        if os.path.isfile(sourceF):
            # 创建目录
            if not os.path.exists(targetDir):
                os.makedirs(targetDir)
                # 文件不存在，或者存在但是大小不同，覆盖
            if not os.path.exists(targetF) or (os.path.exists(targetF) and
                                               (os.path.getsize(targetF) != os.path.getsize(sourceF))):
                open(targetF, "wb").write(open(sourceF, "rb").read())
            else:
                pass
        elif os.path.isdir(sourceF):
            if not os.path.exists(targetF):
                os.makedirs(targetF)
            copy_files(sourceF, targetF)


def get_harbor_tag(recog_type="seg", types="kd-recog"):
    if types == "kd-recog":
        url = "http://10.11.5.137:80/api/repositories/{}%2F{}/tags".format(types, recog_type)
    else:
        url = "http://10.11.5.137:80/api/repositories/{}%2Flane/tags".format(types)
    resp = requests.get(url)
    data = resp.content
    json_data = json.loads(data)
    li = []
    for i in json_data:
        di = {}
        di["name"] = i["name"]
        di["time"] = i["created"]
        li.append(di)
    ddd = sorted(li, key=itemgetter("time"), reverse=True)
    result_li = []
    for i in ddd:
        result_li.append(i["name"])
    return result_li


if __name__ == '__main__':
    get_global_conf("ss", "ss")
