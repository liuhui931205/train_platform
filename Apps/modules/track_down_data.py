# -*-coding:utf-8-*-
import multiprocessing
import os
import time
import requests
import numpy as np
import cv2
from flask import current_app
import shutil
from Apps import krs
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client


class DownloadTask(object):

    def __init__(self, trackpointid, exit_flag=False):
        self.trackPointId = trackpointid
        self.exit_flag = exit_flag


class TrackImage(object):

    def __init__(self):
        self.total_count = 0
        self.krs = krs

    def download_image(self, download_queue, dest):
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

            try:
                url1 = self.krs + "image/get?trackPointId={}&type=00&seq=004&imageType=jpg".format(trackpointid)

                res_data = requests.get(url=url1)
                content_type = res_data.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    current_app.logger.warning("Download  from failed:%s" % url1)

                else:
                    origin_image_data = res_data.content
                    _image0 = np.asarray(bytearray(origin_image_data), dtype="uint8")
                    origin_image = cv2.imdecode(_image0, cv2.IMREAD_COLOR)

                    width = origin_image.shape[1]
                    height = origin_image.shape[0]

                    blank_image = np.zeros((height, width, 3), np.uint8)
                    blank_image[0:height, 0:width] = (255, 255, 255)
                    blank_image[0:height, 0:width] = origin_image

                    img_array = cv2.imencode('.jpg', blank_image)
                    img_data = img_array[1]
                    image_data = img_data.tostring()
                    dir_name = '{}_00_004.jpg'.format(trackpointid)

                    with open(os.path.join(dest, dir_name), "wb") as f:
                        f.write(image_data)
            except Exception as e:
                current_app.logger.error("Download from failed:")


def track_ponit_down(dir_name, trackpointids):
    dir_name = os.path.join("/data/deeplearning/train_platform/eva_sour_data/s_data", dir_name)
    track_handler = TrackImage()
    manager = multiprocessing.Manager()
    download_queue = manager.Queue()

    for i in trackpointids:
        download_task = DownloadTask(trackpointid=i)
        download_queue.put(download_task)

    for x in range(16):
        download_task = DownloadTask(trackpointid=None, exit_flag=True)
        download_queue.put(download_task)
    download_procs = []

    for x in range(16):
        download_proc = multiprocessing.Process(target=track_handler.download_image, args=(download_queue, dir_name))
        # download_proc.daemon = True
        download_procs.append(download_proc)

    for proc in download_procs:
        proc.start()

    for proc in download_procs:
        proc.join()

    final_do(dir_name)


def final_do(dir_name):
    data_name = os.path.basename(dir_name)
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    dirs = dest_sftp.listdir("/data/deeplearning/liuhui/train_platform/eva_sour_data/s_data")
    dest_path = os.path.join("/data/deeplearning/liuhui/train_platform/eva_sour_data/s_data",data_name)
    if data_name not in dirs:
        dest_sftp.mkdir(dest_path)
    files = os.listdir(dir_name)
    for i in files:
        dest_sftp.put(os.path.join(dir_name, i), os.path.join(dest_path, i))

    dirs = dest_sftp.listdir("/data/deeplearning/dataset/training/data/images")
    dest_path = os.path.join("/data/deeplearning/dataset/training/data/images", data_name)
    if data_name not in dirs:
        dest_sftp.mkdir(dest_path)
    files = os.listdir(dir_name)
    for i in files:
        dest_sftp.put(os.path.join(dir_name, i), os.path.join(dest_path, i))



    # dest_scp, dest_sftp, dest_ssh = client(id="host")
    # path = os.path.dirname(dir_name)
    # dirs = dest_sftp.listdir(path)
    # data_name = os.path.basename(dir_name)
    # if data_name not in dirs:
    #     dest_sftp.mkdir(dir_name)
    # files = os.listdir(dir_name)
    # for i in files:
    #     dest_sftp.put(os.path.join(dir_name, i), os.path.join(dir_name, i))
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()
    shutil.rmtree(dir_name)
