from train_platform.celery import app
from utils.scp_server import client
from utils.util import upload, TRAIN_DATA
import time
import datetime
import os
from stat import S_ISDIR
from .models import DataProHandle
import logging
logger = logging.getLogger("django")


@app.task(bind=True)
def pro_data(self, host, docker, cmd, input_path, output_path, conf_path, task_id):

    run_cmd = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} {}".format(
        docker, cmd.format(input_path, output_path, conf_path))
    print(run_cmd)
    try:
        dest_scp, dest_sftp, dest_ssh = client(ids=host)
        stdin, stdout, stderr = dest_ssh.exec_command(run_cmd)

        if stdout.readlines()[-1].strip() == "0":

            obj = DataProHandle.objects.get(task_id=task_id)
            obj.status = "success"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()
            cmd = "rm -r {} && rm -r {}".format(conf_path, input_path)
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()

            d = upload(host, output_path, TRAIN_DATA)
            print(d)
            if not d:
                dest_scp, dest_sftp, dest_ssh = client(ids=host, types="docker")
                stdin, stdout, stderr = dest_ssh.exec_command(cmd)
                cmd = "rm -r {} && rm -r {}".format(output_path, output_path + ".tar.gz")
                stdin1, stdout1, stderr1 = dest_ssh.exec_command(cmd)
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
                return
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()

        cmd = "rm -r {} && rm -r {}".format(conf_path, input_path)
        stdin, stdout, stderr = dest_ssh.exec_command(cmd)
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        obj = DataProHandle.objects.get(task_id=task_id)
        obj.status = "failed"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        return
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())


def test_pro_data(host, docker, cmd, input_path, output_path, conf_path, task_id):

    run_cmd = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning {} {}".format(
        docker, cmd.format(input_path, output_path, conf_path))
    try:
        dest_scp, dest_sftp, dest_ssh = client(ids=host)
        stdin, stdout, stderr = dest_ssh.exec_command(run_cmd)

        if stdout.readlines()[-1].strip() == "0":
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            obj = DataProHandle.objects.get(task_id=task_id)
            obj.status = "success"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()
            return
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        obj = DataProHandle.objects.get(task_id=task_id)
        obj.status = "failed"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        return
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())

@app.task
def clear_data():
    filePath = "/data/deeplearning/train_platform/tmp"
    dest_scp, dest_sftp, dest_ssh = client(ids="165", types="docker")
    now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    now_time = datetime.datetime.strptime(now_time, '%Y-%m-%d %H:%M')
    li = dest_sftp.listdir(filePath)
    for i in li:
        d = dest_sftp.stat(os.path.join(filePath, i))
        timeStruct = datetime.datetime.fromtimestamp(d.st_mtime)
        show_time = datetime.datetime.strftime(timeStruct, '%Y-%m-%d %H:%M')
        show_time = datetime.datetime.strptime(show_time, '%Y-%m-%d %H:%M')
        if (now_time - show_time).days > 15:
            if S_ISDIR(d.st_mode):
                stdin, stdout, stderr = dest_ssh.exec_command("rm -rf {}".format(os.path.join(filePath, i)))
                print(stderr.read())
            else:
                dest_sftp.remove(os.path.join(filePath, i))
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()


if __name__ == '__main__':
    clear_data()
    # test_pro_data("165", "kd-bd02.kuandeng.com/kd-recog/seg:v3.0.5", "bash /opt/kd-seg/tool/preprocess.sh {} {} {}",
    #               "/data/deeplearning/train_platform/src_data/train_data/seg/lane-aug-20200214",
    #               "/data/deeplearning/train_platform/train_data/seg_test",
    #               "/data/deeplearning/train_platform/tmp/DataProHandle_1582610907", "DataProHandle_1582610907")
