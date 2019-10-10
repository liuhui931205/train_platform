# -*-coding:utf-8-*-
import time
from .base_linedown import BaseLineDownTask
import multiprocessing
from Apps.libs.line_downloads.linedown import DownloadTask, TrackImage
import requests
import json
from flask import current_app
import shutil
from Apps.utils.copy_all import copyFiles
from Apps import kms
import os
from Apps.utils.client import client


class LineTasks(BaseLineDownTask):
    def __init__(self):
        super(LineTasks, self).__init__()
        self.error_code = 1
        self.message = 'Linedownload start'
        self.task_id = ''
        self.status = ''
        self.lock = multiprocessing.Lock()

    def create(self, task_id, taskid_start, taskid_end, dest, status):

        self.task_id = task_id
        self.start(task_id, taskid_start, taskid_end, dest, status)
        pro = multiprocessing.Process(target=self.create_async,
                                      args=(taskid_start, taskid_end, dest))
        pro.start()
        self.error_code = 1
        self.message = 'line download start '

    def create_async(self, taskid_start, taskid_end, dest):
        track_point_id = []
        batchs = []
        track_id = []
        start = int(taskid_start)
        end = int(taskid_end) + 1
        try:
            for packid in range(start, end):
                url = kms + 'task/getMarkTagResult?pacId={}'.format(packid)
                req_data = requests.get(url=url)
                req_data = json.loads(req_data.text)
                result = req_data["result"]['node']
                if len(result) > 0:
                    for i in result:
                        h = None
                        for j in i['tag']:
                            if j["k"] == "handle":
                                h = j['v']
                            elif j['k'] == "TRACKPOINTID":
                                t = j['v']
                            elif j['k'] == "BATCH":
                                b = j["v"]
                            elif j['k'] == "TRACKID":
                                r = j["v"]
                        if h == "true":
                            track_point_id.append(t)
                            batchs.append(b)
                            track_id.append(r)

            total_count = len(track_point_id)

            track_handler = TrackImage()
            manager = multiprocessing.Manager()
            download_queue = manager.Queue()
            count_queue = manager.Queue()
            for i in range(total_count):
                download_task = DownloadTask(trackpointid=track_point_id[i],batch=batchs[i],trackid=track_id[i])
                download_queue.put(download_task)

            for x in range(32):
                download_task = DownloadTask(trackpointid=None,batch=None,trackid=None, exit_flag=True)
                download_queue.put(download_task)
            download_procs = []

            for x in range(32):
                download_proc = multiprocessing.Process(target=track_handler.download_image,
                                                        args=(download_queue, count_queue, dest, self.lock))
                # download_proc.daemon = True
                download_procs.append(download_proc)

            for proc in download_procs:
                proc.start()
            count = multiprocessing.Process(target=self.update_task, args=(count_queue, total_count, dest))
            count.start()

            for proc in download_procs:
                proc.join()
            count.join()
        except Exception as e:
            current_app.logger.error(e)

    def update_task(self, count_queue, total_count, dest):

        while True:
            if count_queue.qsize() != 0:
                plan = (int(count_queue.qsize()) / (int(total_count) * 1.00)) * 100
                if plan != 100:
                    self.status = plan
                    value = self.callback(self.task_id, self.status)
                else:
                    self.status = 'completed!'
                    value = self.callback(self.task_id, self.status)
                if value:
                    current_app.logger.info('---------------finished---------------')
                    #
                    time.sleep(5)


                    dest_scp, dest_sftp, dest_ssh = client(id="host")
                    # path = os.path.dirname(dest)
                    dirs = dest_sftp.listdir("/data/deeplearning/dataset/training/data/released")
                    data_name = os.path.basename(dest)
                    data0_dest = os.path.join("/data/deeplearning/dataset/training/data/released", data_name)
                    if data_name not in dirs:
                        dest_sftp.mkdir(data0_dest)
                    files = os.listdir(dest)
                    for i in files:
                        try:
                            di = dest_sftp.listdir(os.path.join(dest, i))
                            if i in di:
                                fi = os.listdir(os.path.join(dest, i))
                                for j in fi:
                                    dest_sftp.put(os.path.join(dest, i, j), os.path.join(data0_dest, i, j))
                            else:
                                dest_scp.put(os.path.join(dest, i), os.path.join(data0_dest, i), recursive=True)
                        except IOError as e:
                            dest_scp.put(os.path.join(dest, i), os.path.join(data0_dest, i), recursive=True)
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                    shutil.rmtree(dest)

                    break
            time.sleep(5)



