from train_platform.celery import app
from utils.scp_server import client
from utils.util import TRAIN_CONF_PATH
from billiard.exceptions import Terminated
import time
from .models import TrainTask
import logging
logger = logging.getLogger("django")


@app.task(bind=True, throws=(Terminated,))
def run_train(self, task_name, docker, host_id, gpu_id, weight, task_id, input_path, output_path, cmd):
    conf_path = TRAIN_CONF_PATH.format(task_name)
    run_cmd = "nvidia-docker run --rm -v /data/deeplearning:/data/deeplearning --name {} {} {}".format(
        task_id, docker, cmd.format(input_path, output_path, weight, gpu_id, conf_path))
    print(run_cmd)
    try:
        dest_scp, dest_sftp, dest_ssh = client(ids=host_id)
        stdin, stdout, stderr = dest_ssh.exec_command(run_cmd)

        if stdout.readlines()[-1].strip() == "0":
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
            obj = TrainTask.objects.get(task_id=task_id)
            obj.status = "success"
            obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
            obj.save()
            return
        dest_scp.close()
        dest_sftp.close()
        dest_ssh.close()
        obj = TrainTask.objects.get(task_id=task_id)
        obj.status = "failed"
        obj.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        obj.save()
        return
    except Exception as e:
        import traceback
        logger.error(traceback.format_exc())
