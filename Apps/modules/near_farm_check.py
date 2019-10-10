# -*-coding:utf-8-*-
import requests
import json
from flask import current_app
from config import json_path, data_sour_train
import os
from Apps.models import TrainTask
import multiprocessing
from Apps.libs.auto_sam_main import TrackImage, Task, DownloadTask
import time
from .base_check import BaseCkeckTask
from Apps import db, mach_id, krs
from Apps.utils.client import client
import shutil
import random


class CheckTasks(BaseCkeckTask):

    def __init__(self):
        super(CheckTasks, self).__init__()
        self.error_code = 1
        self.message = 'Linedownload start'
        self.task_id = ''
        self.status = ''
        self.sele_ratio = None

    def starts(self, task_name, data_li, gpus, task_id, weights_dir, status):
        super(CheckTasks, self).start(task_name, gpus, task_id, weights_dir, status)
        self.task_id = task_id
        self.status = status
        data_dir = os.path.join("/data/deeplearning/check_sele_datas", task_name)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        sele_dir = os.path.join(data_dir, "sele")
        if not os.path.exists(sele_dir):
            os.makedirs(sele_dir)
        data_li = data_li.split(',')
        train_task = db.session.query(TrainTask).filter_by(task_name=weights_dir).first()
        machine_id = train_task.machine_id
        dest = '/data/deeplearning/train_platform/select_data/models/' + weights_dir
        sour = os.path.join(data_sour_train, weights_dir, '/output/models')
        models = []
        if mach_id != machine_id:
            dest_scp, dest_sftp, dest_ssh = client(id=machine_id)
            model_li = dest_sftp.listdir(sour)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in model_li:
                if i.endswith('.params'):
                    models.append(i)
                else:
                    if not os.path.exists(os.path.join(dest, i)):
                        dest_sftp.get(os.path.join(sour, i), os.path.join(dest, i))
            models.sort()
            if not os.path.exists(os.path.join(dest, models[-1])):
                dest_sftp.get(os.path.join(sour, models[-1]), os.path.join(dest, models[-1]))
            model_dir = os.path.join(dest, models[-1])
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
        else:
            model_li = os.listdir(sour)
            if not os.path.exists(dest):
                os.makedirs(dest)
            for i in model_li:
                if i.endswith('.params'):
                    models.append(i)
                else:
                    with open(os.path.join(sour, i), 'r') as f:
                        datas = f.read()
                    with open(os.path.join(dest, i), 'w') as e:
                        e.write(datas)
            models.sort()
            if not os.path.exists(os.path.join(dest, models[-1])):
                with open(os.path.join(sour, models[-1]), 'r') as f:
                    datas = f.read()
                with open(os.path.join(dest, models[-1]), 'w') as e:
                    e.write(datas)
            model_dir = os.path.join(dest, models[-1])
        pro1 = multiprocessing.Process(
            target=self.auto_sele, args=(task_name, data_li, gpus, model_dir, data_dir, sele_dir))
        pro1.start()
        self.error_code = 1
        self.message = 'Check_data start'

    def get_track_point_ids(self, data_li, frame_count=20):
        track_li = []
        for track_point_id in data_li:
            track_point_url = "{}track/point/get?trackPointId={}".format(krs, track_point_id)
            resp = requests.get(track_point_url)
            if resp.status_code == 200:
                track_point_info = json.loads(resp.text)
                if "code" not in track_point_info or track_point_info["code"] != "0":
                    continue
                track_id = track_point_info["result"]["trackId"]

                track_url = "{}track/get?trackId={}".format(krs, track_id)
                resp = requests.get(track_url)
                # logging.info(resp.text)
                if resp.status_code == 200:
                    track_info = json.loads(resp.text)
                    if "code" not in track_info or track_info["code"] != "0":
                        continue
                    track_points = []
                    points = track_info["result"]["pointList"]

                    for point in points:
                        track_points.append(point["trackPointId"])
                    track_points.sort()

                    pt_index = 0
                    for _id in track_points:
                        if _id == track_point_id:
                            break
                        pt_index += 1
                    start_index = end_index = pt_index
                    if pt_index < frame_count:
                        start_index = 0
                    else:
                        start_index = pt_index - frame_count

                    if (pt_index + frame_count) >= len(track_points):
                        end_index = len(track_points)
                    else:
                        end_index = pt_index + frame_count + 1

                    to_download = track_points[start_index:end_index]
                    for i in to_download:
                        if i not in track_li:
                            track_li.append(i)
        return track_li

    def auto_sele(self, task_name, data_li, gpus, weights_dir, data_dir, sele_dir):
        single_count = 2
        cpus = 16
        track_handler = TrackImage()
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        download_queue = manager.Queue()
        count_queue = manager.Queue()
        lock = multiprocessing.Lock()
        base_url = krs
        sel_points = self.get_track_point_ids(data_li)
        self.sele_ratio = len(sel_points)

        for track_point_id in sel_points:
            download_task = DownloadTask(track_point_id=track_point_id)
            download_queue.put(download_task)

        for x in range(cpus):
            download_task = DownloadTask(track_point_id=None, exit_flag=True)
            download_queue.put(download_task)

        download_procs = []
        for x in range(cpus):
            download_proc = multiprocessing.Process(
                target=track_handler.download_image, args=(download_queue, task_queue, base_url, data_dir))
            download_proc.daemon = True
            download_procs.append(download_proc)

        for proc in download_procs:
            proc.start()

        gpu_ids = gpus
        all_process = []

        gpu_ids = gpu_ids.split(",")
        gpu_count = single_count * len(gpu_ids)

        for i in range(gpu_count):
            gpu_id = int(gpu_ids[i / single_count])
            processor = multiprocessing.Process(
                target=track_handler.do, args=(gpu_id, weights_dir, data_dir, sele_dir, task_queue, count_queue))
            all_process.append(processor)
        count = multiprocessing.Process(target=self.update_task, args=(count_queue, task_name, sele_dir))
        count.start()
        for j in range(single_count):
            for i in range(len(gpu_ids)):
                t = all_process[i * single_count + j]
                t.start()
            time.sleep(30)

        for proc in download_procs:
            proc.join()

        for i in range(gpu_count):
            exit_task = Task(image_path=None, exit_flag=True)
            task_queue.put(exit_task)
        count.join()
        for process in all_process:
            process.join()

    def update_task(self, count_queue, task_name, sele_dir):
        while True:
            size = count_queue.qsize()

            if int(size) != int(self.sele_ratio):
                plan = (int(size) / (int(self.sele_ratio) * 1.00)) * 100
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                value = self.callback(self.task_id, self.status)
            if value:
                self.rm_dup_upload(sele_dir, task_name)
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(6)

    def rm_dup_upload(self, output_dir, task_name):
        out_dir = os.path.dirname(output_dir)
        src_file_list = []
        for image_file in os.listdir(output_dir):
            if image_file.endswith('jpg'):
                src_file_list.append(image_file)
        json_data = []
        for chosen_id in src_file_list:
            chosen_id = chosen_id[:-11]
            url = krs + 'track/point/get?trackPointId={}'.format(chosen_id)
            req_data = requests.get(url)
            req_data = json.loads(req_data.text)
            trackid = req_data['result']['trackId']
            dicts = {}
            dicts["TRACKPOINTID"] = chosen_id
            dicts["IMGOPRANGE"] = ""
            dicts["ROADELEMENT"] = ""
            dicts["SOURCE"] = ""
            dicts["AUTHOR"] = ""
            dicts["CITY"] = ""
            dicts["DATAKIND"] = ""
            dicts["ANNOTYPE"] = ""
            dicts["PACKAGEID"] = ""
            dicts["ROADANGLE"] = ""
            dicts["ISREPAIR"] = ""
            dicts["DOWNSUFFIX"] = ""
            dicts["TRACKPOINTSEQ"] = ""
            dicts["TRACKLOCATION"] = ""
            dicts["IMGID"] = ""
            dicts["DOWNTYPE"] = ""
            dicts["BATCH"] = ""
            dicts["SCENE"] = ""
            dicts["EXTEND2"] = ""
            dicts["EXTEND3"] = ""
            dicts["MARKTYPE"] = ""
            dicts["EXTEND4"] = ""
            dicts["WEATHER"] = ""
            dicts["ROADTYPE"] = ""
            dicts["ERRORTYPE"] = ""
            dicts["TRACKID"] = ""
            dicts["EXTEND1"] = ""
            dicts["TASKID"] = ""
            dicts["MARKTASKID"] = ""
            dicts["DATATYPE"] = ""
            dicts["DOWNSEQ"] = ""
            dicts["MARKBY"] = ""
            dicts["SEQ"] = ""
            dicts["MARKTIME"] = ""
            dicts["TRACKID"] = trackid
            dicts["handle"] = ""
            json_data.append(dicts)
        random.shuffle(json_data)
        with open(os.path.join(out_dir, task_name + ".json"), "w") as f:
            json.dump(json_data, f)

        shutil.rmtree(output_dir)
