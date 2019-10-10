# -*-coding:utf-8-*-
import time
from .base_offline import BaseOfflineTask
import multiprocessing
import os
import json
import requests
from flask import current_app
from Apps import krs


class OffTasks(BaseOfflineTask):
    def __init__(self):
        super(OffTasks, self).__init__()
        self.error_code = 1
        self.message = 'Offimport start'
        self.task_id = ''
        self.status = ''

    def create(self, src, dest, task_id, roadelement, source, author, annotype, datakind, city, imgoprange, status):

        self.task_id = task_id
        self.start(task_id, roadelement, source, author, annotype, datakind, city, imgoprange, status)
        pro = multiprocessing.Process(target=self.create_async,
                                      args=(
                                          src, dest, roadelement, source, author, annotype, datakind, city, imgoprange))
        pro.start()
        self.error_code = 1
        self.message = 'Offimport start '

    def create_async(self, src, dest, roadelement, source, author, annotype, datakind, city, imgoprange):
        lists = []
        manager = multiprocessing.Manager()
        count_queue = manager.Value('i', 0)
        task_name = os.path.basename(src)
        # dest_path = os.path.join(dest, task_name)
        dest_path = dest
        value = os.popen("ls -lR " + src + "|grep 'jpg' | wc -l").readlines()[0]
        value = int(value.strip())
        print(value)
        count_queue.value = value
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)
        s = os.listdir(src)
        count = multiprocessing.Process(target=self.update_task, args=(count_queue, value))
        count.start()
        for i in s:
            lists = []
            image_path = os.path.join(src, i)
            img_li = os.listdir(image_path)

            if annotype == "0":
                for j in img_li:
                    # if j.startswith('label-') and j.endswith('.png'):
                    if j.endswith('.jpg'):
                        trackpointid = j[0:-11]
                        url = krs + 'track/point/get?trackPointId={}'.format(trackpointid)
                        req_data = requests.get(url)
                        req_data = json.loads(req_data.text)
                        trackid = req_data['result']['trackId']
                        dicts = {}
                        dicts["TRACKPOINTID"] = trackpointid
                        dicts["IMGOPRANGE"] = imgoprange
                        dicts["ROADELEMENT"] = roadelement
                        dicts["SOURCE"] = source
                        dicts["AUTHOR"] = author
                        dicts["CITY"] = city
                        dicts["DATAKIND"] = datakind
                        dicts["ANNOTYPE"] = annotype
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
                        # open(os.path.join(dest_path, label_name), "wb").write(open(os.path.join(src, i, label), "rb").read())
                        time.sleep(0.05)
                        count_queue.value -= 1
                        lists.append(dicts)
            else:
                for j in img_li:
                    # if j.startswith('label-') and j.endswith('.png'):
                    if j.endswith('.jpg') and not j.startswith('ext-'):
                        trackpointid = j[0:-11]
                        label_name = j[0:-11] + '_70_001.png'
                        label = 'label-' + j[0:-11] + '_00_004.png'
                        try:
                            url = krs + 'track/point/get?trackPointId={}'.format(
                                trackpointid)
                            req_data = requests.get(url)
                            req_data = json.loads(req_data.text)
                            trackid = req_data['result']['trackId']
                        except Exception as e:
                            print(e)
                            count_queue.value -= 1
                            continue
                        dicts = {}
                        dicts["TRACKPOINTID"] = trackpointid
                        dicts["IMGOPRANGE"] = imgoprange
                        dicts["ROADELEMENT"] = roadelement
                        dicts["SOURCE"] = source
                        dicts["AUTHOR"] = author
                        dicts["CITY"] = city
                        dicts["DATAKIND"] = datakind
                        dicts["ANNOTYPE"] = annotype
                        dicts["PACKAGEID"] = ""
                        dicts["ROADANGLE"] = ""
                        dicts["ISREPAIR"] = ""
                        dicts["DOWNSUFFIX"] = ""
                        dicts["TRACKPOINTSEQ"] = ""
                        dicts["TRACKLOCATION"] = ""
                        dicts["IMGID"] = ""
                        dicts["DOWNTYPE"] = ""
                        dicts["BATCH"] = "0_19"
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
                        try:
                            open(os.path.join(dest_path, label_name), "wb").write(
                                open(os.path.join(src, i, label), "rb").read())
                            time.sleep(0.05)

                            if dicts not in lists:
                                lists.append(dicts)
                        except Exception as e:
                            print(e)
                        count_queue.value -= 1

            json.dump(lists, open(os.path.join(dest_path, i + '.json'), 'w'))
        count.join()

    def update_task(self, count_queue, value):
        while True:
            plan = (1 - (count_queue.value / (value * 1.0))) * 100.00
            if plan != 100:
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                value = self.callback(self.task_id, self.status)
            if value:
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(3)
