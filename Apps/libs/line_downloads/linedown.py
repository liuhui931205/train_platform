# -*- coding:utf-8 -*-
import os
import requests
import time
import cv2
import numpy as np
from flask import current_app
from Apps import mark_get


class DownloadTask(object):

    def __init__(self, trackpointid, batch,trackid, exit_flag=False):
        self.trackPointId = trackpointid
        self.batch = batch
        self.trackid = trackid
        self.exit_flag = exit_flag


class TrackImage(object):

    def __init__(self):
        self.total_count = 0

    @staticmethod
    def download_image(download_queue, count_queue, dest, lock):
        if not os.path.exists(dest):
            os.makedirs(dest)
        while True:
            if download_queue.empty():
                time.sleep(3)

            download_task = download_queue.get()
            if not isinstance(download_task, DownloadTask):
                break

            if download_task.exit_flag:
                break

            trackpointid = download_task.trackPointId
            batch = download_task.batch
            trackid = download_task.trackid
            try:
                url1 = mark_get + "mark_image/get?trackPointId={}&type=00&seq=004&imageType=jpg&trackId={}".format(trackpointid,trackid)
                url2 = mark_get + "mark_image/get?trackPointId={}&type=70&seq=001&imageType=png&batch={}&trackId={}".format(
                    trackpointid, batch, trackid)
                s = [url2, url1]
                lock.acquire()
                count_queue.put(1)
                lock.release()
                for i in s:
                    res_data = requests.get(url=i, timeout=1000)
                    content_type = res_data.headers['Content-Type']
                    if not str(content_type).startswith("image"):
                        current_app.logger.warning("Download  from failed:%s" % i)
                        break
                    else:
                        origin_image_data = res_data.content
                        _image0 = np.asarray(bytearray(origin_image_data), dtype="uint8")
                        origin_image = cv2.imdecode(_image0, cv2.IMREAD_COLOR)

                        width = origin_image.shape[1]
                        height = origin_image.shape[0]

                        blank_image = np.zeros((height, width, 3), np.uint8)
                        blank_image[0:height, 0:width] = (255, 255, 255)
                        blank_image[0:height, 0:width] = origin_image

                        if s[1] == i:
                            img_array = cv2.imencode('.jpg', blank_image)
                            img_data = img_array[1]
                            image_data = img_data.tostring()
                            dir_name = '{}_00_004.jpg'.format(trackpointid)
                        else:
                            img_array = cv2.imencode('.png', blank_image)
                            img_data = img_array[1]
                            image_data = img_data.tostring()
                            dir_name = 'label-{}_00_004.png'.format(trackpointid)
                        if not os.path.exists(os.path.join(dest, '1')):
                            os.makedirs(os.path.join(dest, '1'))
                        with open(os.path.join(dest, '1', dir_name), "wb") as f:
                            f.write(image_data)

            except Exception as e:
                current_app.logger.error("Download from failed:")
