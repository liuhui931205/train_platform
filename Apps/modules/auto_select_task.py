# -*-coding:utf-8-*-
import json
import os
import random
import time
import requests
import multiprocessing
from flask import current_app
from Apps.libs.auto_sam_main import DownloadSamTask, TrackSamImage, TrackImage, DownloadTask, Task
from Apps.models import TrainTask
from Apps import db, mach_id, krs
from Apps.utils.client import client
from .base_select import BaseSelectTask
import shutil
import re
from Apps.utils.copy_all import copyFiles


class AutoSelectTask(BaseSelectTask):
    def __init__(self):
        super(AutoSelectTask, self).__init__()
        self.error_code = 1
        self.message = 'Autosele start'
        self.task_id = ''
        self.status = ''
        self.sele_ratio = ''

    def start(self, output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status, isshuffle, task_type,
              task_file):
        super(AutoSelectTask, self).start(output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status,
                                          isshuffle, task_type, task_file)

        self.task_id = task_id
        self.status = status
        self.sele_ratio = sele_ratio

        # 模型下载或复制到固定位置
        train_task = db.session.query(TrainTask).filter_by(task_name=weights_dir).first()
        machine_id = train_task.machine_id
        dest = '/data/deeplearning/train_platform/select_data/models/' + weights_dir
        sour = '/data/deeplearning/train_platform/train_task/' + weights_dir + '/output/models'
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

        if task_file is not None:
            track_dir = os.path.join(output_dir, 'track_file')
            if not os.path.exists(track_dir):
                os.makedirs(track_dir)
            now_track_file = self.get_track_ids(task_file, track_dir + '/track.txt')
            pro1 = multiprocessing.Process(target=self.auto_sele_trackid,
                                           args=(output_dir, gpus, now_track_file, sele_ratio, model_dir, isshuffle))
            # pro1 = multiprocessing.Process(target=self.auto_sele_trackid,
            #                                args=(output_dir, gpus, now_track_file, sele_ratio, weights_dir, isshuffle))
            pro1.start()
        else:
            if not os.path.exists(track_file):
                current_app.logger.warning("track_file[{}] is not exist!".format(track_file))
            else:
                pro = multiprocessing.Process(target=self.auto_sele_trackid,
                                              args=(output_dir, gpus, track_file, sele_ratio, weights_dir, isshuffle))
                pro.start()
        self.error_code = 1
        self.message = 'Autosele start'

    def auto_sele_trackid(self, output_dir, gpus, track_file, sele_ratio, weights_dir, isshuffle):
        base_url = krs
        single_count = 4
        cpus = 32
        data_dir = output_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        sele_dir = os.path.join(output_dir, "sele")
        if not os.path.exists(sele_dir):
            os.makedirs(sele_dir)

        track_handler = TrackImage()
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        download_queue = manager.Queue()
        count_queue = manager.Queue()
        image_file_list = []
        if not track_file:
            for image_file in os.listdir(data_dir):
                if image_file.endswith("jpg"):
                    image_file_list.append(image_file)

            seled_list = []
            for image_file in os.listdir(sele_dir):
                if image_file.endswith("jpg"):
                    seled_list.append(image_file)

            image_file_list = list(set(image_file_list).difference(set(seled_list)))

            cursor = 0
            cursor_to = len(image_file_list)
            for i in range(cursor_to - cursor):
                image_file = image_file_list[cursor + i]

                new_task = Task(image_path=image_file)
                task_queue.put(new_task)
        else:
            if not os.path.exists(track_file):
                current_app.logger.warning("track_file[{}] is not exist!".format(track_file))
                exit(0)

            track_ids = []
            with open(track_file, "r") as f:
                track_id = f.readline()
                track_id = track_id.strip()
                while track_id:
                    track_ids.append(track_id)
                    track_id = f.readline()
                    track_id = track_id.strip()

            total_count = 0
            track_points = []
            for track_id in track_ids:
                track_id = track_id.strip()
                print(track_id)

                url = base_url + "/track/get"

                try:
                    point_data = list()

                    res = requests.post(url=url, data={'trackId': track_id})
                    track_info = res.text

                    track_data = json.loads(track_info)
                    code = track_data["code"]

                    if code != "0":
                        continue

                    points = track_data["result"]["pointList"]
                    current_app.logger.debug("trackPointCount:{}".format(len(points)))
                    for point in points:
                        point_data.append(point)
                    task_count = len(points)

                    total_count += task_count
                    for point in point_data:
                        track_point_id = point["trackPointId"]
                        track_points.append(track_point_id)
                except Exception as e:
                    current_app.logger.error(e.args[0])
                    continue

            if isshuffle:
                random.seed(1000)
                random.shuffle(track_points)

            if sele_ratio > 1.0:
                sel_count = int(sele_ratio)
            else:
                sel_count = int(sele_ratio * total_count)
            sel_points = track_points[:sel_count]
            current_app.logger.debug("process image count:{}".format(sel_count))

            for track_point_id in sel_points:
                download_task = DownloadTask(track_point_id=track_point_id)
                download_queue.put(download_task)

            for x in range(cpus):
                download_task = DownloadTask(track_point_id=None, exit_flag=True)
                download_queue.put(download_task)

        download_procs = []
        for x in range(cpus):
            download_proc = multiprocessing.Process(target=track_handler.download_image,
                                                    args=(download_queue, task_queue, base_url, data_dir))
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
            processor = multiprocessing.Process(target=track_handler.do,
                                                args=(gpu_id, weights_dir, data_dir, sele_dir, task_queue, count_queue))
            all_process.append(processor)
        count = multiprocessing.Process(target=self.update_task, args=(count_queue, output_dir))
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

    def get_track_ids(self, task_file, out_file,
                      url=krs + 'track/get/trackIds/bytask?taskId={}'):
        taskIds = []
        task_ids = task_file.split(',')
        for task in task_ids:
            task = task.strip()
            taskIds.append(task)
        # with open(task_file, "r") as f:
        # 	line_str = f.readline()
        # 	while line_str:
        # 		line_str = line_str.strip()
        # 		taskIds.append(line_str)
        # 		line_str = f.readline()
        with open(out_file, "w") as f:
            for task_id in taskIds:
                track_url = url.format(task_id)

                resp = requests.get(track_url)
                track_info = json.loads(resp.text)

                track_info = track_info["result"]

                for track_id in track_info:
                    f.write("{}\n".format(track_id))
        return out_file

    def update_task(self, count_queue, output_dir):
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
                self.rm_dup_upload(output_dir)
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(6)

    def rm_dup_upload(self, output_dir):
        dest_dir = "/data/deeplearning/auto_sele_datas"
        task = os.path.basename(os.path.dirname(output_dir))
        name = os.path.basename(output_dir)
        s_name = name.split('.')[0]
        src_dir = os.path.join(output_dir, 'sele')
        src_file_list = []
        for image_file in os.listdir(src_dir):
            if image_file.endswith('jpg'):
                src_file_list.append(image_file)

        for v in range(1, 5):
            rmlistfile = []
            task_path = os.path.dirname(output_dir)
            path_li = os.listdir(task_path)
            name_li = []
            for i in path_li:
                if re.match(r'.*v\d+', i):
                    name_li.append(os.path.join(task_path, i))
            if name_li:
                for i in name_li:
                    li = os.listdir(i)
                    for j in li:
                        if j.endswith('.jpg'):
                            rmlistfile.append(j)
            goal_list = list(set(src_file_list).difference(set(rmlistfile)))
            random.shuffle(goal_list)
            sam_list = goal_list[:1000]
            p_name = s_name + str(1000) + ".sele.v" + str(v)
            out_dir = os.path.join(task_path, p_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            json_data = []
            for chosen_id in sam_list:
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
            with open(os.path.join(out_dir, "task_list.json"), "w") as f:
                json.dump(json_data, f)

            for img_file in sam_list:
                shutil.copy(os.path.join(src_dir, img_file), os.path.join(out_dir, img_file))

            if mach_id != '13':
                dest_scp, dest_sftp, dest_ssh = client(id="13")
                files = dest_sftp.listdir(dest_dir)
                dest = os.path.join(dest_dir, task)
                if task not in files:
                    dest_sftp.mkdir(dest)
                dest_scp.put(out_dir, dest, recursive=True)
                dest_scp.close()
                dest_sftp.close()
                dest_ssh.close()
            else:
                pa = os.path.basename(out_dir)
                files = os.listdir(dest_dir)
                dests = os.path.join(dest_dir, task, pa)
                if task not in files:
                    os.makedirs(dests)
                copyFiles(out_dir, dests)


class AutoSamTask(BaseSelectTask):
    def __init__(self):
        super(AutoSamTask, self).__init__()
        self.error_code = 1
        self.message = 'Autosam start'
        self.task_id = ''
        self.status = ''
        self.sele_ratio = ''
        self.lock = multiprocessing.Lock()

    def start(self, output_dir, gpus, sele_ratio, weights_dir, track_file, task_id, status, isshuffle, task_type,
              task_file):
        super(AutoSamTask, self).start(output_dir, gpus, sele_ratio, weights_dir, track_file, task_id,
                                       status,
                                       isshuffle, task_type, task_file)
        self.task_id = task_id
        self.status = status
        self.sele_ratio = sele_ratio
        self.error_code = 1
        self.message = 'Autosam start'
        if task_file:
            track_dir = os.path.join(output_dir, 'track_file')
            if not os.path.exists(track_dir):
                os.makedirs(track_dir)
            now_track_file = self.get_track_ids(task_file, track_dir + '/track.txt')
            pro1 = multiprocessing.Process(target=self.sam_main,
                                           args=(output_dir, sele_ratio, isshuffle, now_track_file))
            pro1.start()
        else:
            if not os.path.exists(track_file):
                current_app.logger.warning("track_file[{}] is not exist!".format(track_file))
            else:
                pro2 = multiprocessing.Process(target=self.sam_main,
                                               args=(output_dir, sele_ratio, isshuffle, track_file))
                pro2.start()

    def sam_main(self, output_dir, ratio, isshuffle, track_file):
        # todo
        base_url = krs
        time0 = time.time()
        track_ids = []
        with open(track_file, "r") as f:
            track_id = f.readline()
            track_id = track_id.strip()
            while track_id:
                track_ids.append(track_id)
                track_id = f.readline()
                track_id = track_id.strip()

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_count = 0
        track_points = []
        for track_id in track_ids:
            track_id = track_id.strip()
            print(track_id)

            url = base_url + "/track/get"

            try:
                point_data = list()

                res = requests.post(url=url, data={'trackId': track_id})
                track_info = res.text

                track_data = json.loads(track_info)
                code = track_data["code"]

                if code != "0":
                    continue

                points = track_data["result"]["pointList"]
                current_app.logger.debug("trackPointCount:{}".format(len(points)))
                for point in points:
                    point_data.append(point)
                task_count = len(points)

                total_count += task_count
                for point in point_data:
                    track_point_id = point["trackPointId"]
                    track_points.append(track_point_id)
            except Exception as e:
                current_app.logger.error(e.args[0])
                continue

        if isshuffle:
            random.seed(1000)
            random.shuffle(track_points)

        if ratio > 1.0:
            sel_count = int(ratio)
        else:
            sel_count = int(ratio * total_count)
        sel_points = track_points[:sel_count]
        current_app.logger.debug("process image count:{}".format(sel_count))
        # write json
        json_data = []
        for chosen_id in sel_points:
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
        with open(os.path.join(output_dir, "task_list.json"), "w") as f:
            json.dump(json_data, f)

        track_handler = TrackSamImage()
        manager = multiprocessing.Manager()
        download_queue = manager.Queue()
        count_queue = manager.Queue()

        for track_point_id in sel_points:
            download_task = DownloadSamTask(track_point_id=track_point_id)
            download_queue.put(download_task)
            count_queue.put(1)

        for x in range(32):
            download_task = DownloadSamTask(track_point_id=None, exit_flag=True)
            download_queue.put(download_task)
        download_procs = []
        for x in range(32):
            download_proc = multiprocessing.Process(target=track_handler.download_image,
                                                    args=(download_queue, count_queue, base_url, output_dir, self.lock))
            download_proc.daemon = True
            download_procs.append(download_proc)

        for proc in download_procs:
            proc.start()
        count = multiprocessing.Process(target=self.update_task, args=(count_queue, output_dir))
        count.start()

        for proc in download_procs:
            proc.join()
        count.join()

    def get_track_ids(self, task_file, out_file,
                      url=krs + 'track/get/trackIds/bytask?taskId={}'):
        taskIds = []
        task_ids = task_file.split(',')
        for task in task_ids:
            task = task.strip()
            taskIds.append(task)

        # with open(task_file, "r") as f:
        # 	line_str = f.readline()
        # 	while line_str:
        # 		line_str = line_str.strip()
        # 		taskIds.append(line_str)
        # 		line_str = f.readline()
        with open(out_file, "w") as f:
            for task_id in taskIds:
                track_url = url.format(task_id)

                resp = requests.get(track_url)
                track_info = json.loads(resp.text)

                track_info = track_info["result"]

                for track_id in track_info:
                    f.write("{}\n".format(track_id))
        return out_file

    def update_task(self, count_queue, output_dir):
        while True:
            size = count_queue.qsize()
            if int(size) != 0:
                plan = 100.0 - (int(size) / (int(self.sele_ratio) * 1.00)) * 100
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                value = self.callback(self.task_id, self.status)
            if value:
                self.upload(output_dir)
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(6)

    def upload(self, output_dir):
        dest_dir = "/data/deeplearning/auto_sele_datas"
        task = os.path.basename(os.path.dirname(output_dir))
        name = os.path.basename(output_dir)
        if mach_id != '13':
            dest_scp, dest_sftp, dest_ssh = client(id="13")
            files = dest_sftp.listdir(dest_dir)
            dest_dir = os.path.join(dest_dir, task)
            if task not in files:
                dest_sftp.mkdir(dest_dir)
            dest_scp.put(output_dir, dest_dir, recursive=True)
            dest_scp.close()
            dest_sftp.close()
            dest_ssh.close()
        else:
            files = os.listdir(dest_dir)
            dest_dir = os.path.join(dest_dir, task, name)
            if task not in files:
                os.makedirs(dest_dir)
            copyFiles(output_dir, dest_dir)
