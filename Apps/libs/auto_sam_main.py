#!/usr/bin/env python
# -*-coding:utf-8-*-


import os
import shutil
import time
from flask import current_app
import cv2
import numpy as np
import requests
from Apps.libs.auto_sele_base_pred.c_model import ModelResNetRoad


class DownloadSamTask(object):
    def __init__(self, track_point_id, exit_flag=False):
        self.track_point_id = track_point_id
        self.exit_flag = exit_flag


class TrackSamImage(object):
    def __init__(self):
        self.total_count = 0

    @staticmethod
    def download_image(download_queue, count_queue, base_url, dest_dir, lock):
        while True:
            if download_queue.empty():
                time.sleep(3)

            download_task = download_queue.get()
            if not isinstance(download_task, DownloadSamTask):
                break

            if download_task.exit_flag:
                break

            track_point_id = download_task.track_point_id
            url = base_url + "image/get"
            data = {
                "trackPointId": track_point_id,
                "type": "00",
                "seq": "004",
                "imageType": "jpg"
            }
            url_str = ""
            try:
                res_data = requests.post(url=url, data=data)

                url_str = "{}?trackPointId={}&type=00&seq=004&imageType=jpg".format(url, track_point_id)
                content_type = res_data.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    current_app.logger.error(
                        "Download {} from {} failed:{}".format(track_point_id, url_str, res_data.text.encode("UTF-8")))
                else:
                    lock.acquire()
                    image_data = res_data.content
                    image_name = "{}_00_004.jpg".format(track_point_id)
                    image_path = os.path.join(dest_dir, image_name)
                    with open(image_path, "wb") as f:
                        f.write(image_data)

                    count_queue.get()
                    lock.release()
            except Exception as e:
                current_app.logger.error("Download {} from {} failed:{}".format(track_point_id, url_str, repr(e)))


class DownloadTask(object):
    def __init__(self, track_point_id, exit_flag=False):
        self.track_point_id = track_point_id
        self.exit_flag = exit_flag


class Task(object):
    def __init__(self, image_path=None, exit_flag=False):
        self._image_path = image_path
        self._exit_flag = exit_flag


class TrackImage(object):
    def __init__(self):
        self.total_count = 0

    @staticmethod
    def download_image(download_queue, task_queue, base_url, dest_dir):
        while True:
            if download_queue.empty():
                time.sleep(3)

            download_task = download_queue.get()
            if not isinstance(download_task, DownloadTask):
                break

            if download_task.exit_flag:
                break

            track_point_id = download_task.track_point_id
            # track_handler.add_task(_track_dir=dest_dir, track_point_id=track_point_id)
            url = base_url + "image/get"
            data = {
                "trackPointId": track_point_id,
                "type": "00",
                "seq": "004",
                "imageType": "jpg"
            }
            url_str = ""
            try:
                res_data = requests.post(url=url, data=data)

                url_str = "{}?trackPointId={}&type=00&seq=004&imageType=jpg".format(url, track_point_id)
                content_type = res_data.headers['Content-Type']
                if not str(content_type).startswith("image"):
                    current_app.logger.error(
                        "Download {} from {} failed:{}".format(track_point_id, url_str, res_data.text.encode("UTF-8")))
                else:
                    image_data = res_data.content
                    image_name = "{}_00_004.jpg".format(track_point_id)
                    image_path = os.path.join(dest_dir, image_name)
                    with open(image_path, "wb") as f:
                        f.write(image_data)
                    check_task = Task(image_path=image_path, exit_flag=False)
                    task_queue.put(check_task)
            except Exception as e:
                current_app.logger.error("Download {} from {} failed:{}".format(track_point_id, url_str, repr(e)))

    def get_idOI(self, data_dir, sele_dir, image_list, pred_label, confidence, idOI=[0, 1, 2, 4, 6]):
        t0 = time.time()
        print('gpu_upsampling time {}'.format(time.time() - t0))
        h, w = pred_label.shape
        h0 = h // 4

        sec_pred = pred_label[h0:-10, :]
        all_pixel_roi = np.where(
            (sec_pred == 0) | (sec_pred == 1) | (sec_pred == 2) | (sec_pred == 4) | (sec_pred == 6))

        all_pixel_xy = np.array(zip(all_pixel_roi[0], all_pixel_roi[1]))  # [(1, 4), (3, 7), (5, 8)]

        if len(all_pixel_xy) == 0:
            print("--------------------ignore---------------------------")
            return 0
        else:
            samp_pixel = all_pixel_xy
            for i, j in samp_pixel:
                ij_list = range(1)
                if i in ij_list or j in ij_list:
                    continue
                if i >= h - h0 - 1 - 10 or j >= w - 1:
                    continue
                temp1 = list(np.unique(self.get_roi(i, j, 1, sec_pred)))
                idlist = list(set(temp1).intersection(set(idOI)))

                id_roi = self.get_roi(i, j, 5, sec_pred)
                id_roi1 = self.get_roi(i, j, 9, sec_pred)
                conf = confidence[0][h0:-10, :]
                roi_conf = self.get_roi(i, j, 5, conf)
                sum_pixel = 0
                sum_conf = 0.0
                for k in idOI:
                    sum_pixel += np.sum(id_roi == k)
                    sum_conf += np.sum(roi_conf[id_roi == k])
                ave_conf = sum_conf / float(sum_pixel)
                is_comet = self.get_roi(i, j, 5, conf)

                if ave_conf < 0.35:
                    self.save_result_s(data_dir, sele_dir, image_list, pred_label, confidence)
                    return
                if len(idlist) >= 2 and ave_conf < 0.8:
                    # 求类别最小值
                    num_id = np.ones(max(idlist) + 1)
                    for id in idlist:
                        num_id[id] = np.sum(id_roi == id)
                    numid = sorted(list(num_id))
                    for item in numid:
                        if item == 1:
                            numid.remove(item)
                    try:
                        if np.sum(roi_conf > 0.55) < 25 and numid[0] > 10:
                            self.save_result_s(data_dir, sele_dir, image_list, pred_label, confidence)
                            return
                    except Exception as e:
                        print('img_file : {}'.format(image_list))
                        return
                if len(idlist) == 1 and np.sum(is_comet < 0.55) > 100:
                    num_cont = self.get_num_contours(id_roi1, idlist)
                    if num_cont > 1:
                        self.save_result_s(data_dir, sele_dir, image_list, pred_label, confidence)
                        return

    def get_num_contours(self, is_comet_id, idlist):
        is_comet_id_temp = np.zeros_like(is_comet_id)

        is_comet_id_temp[np.where((is_comet_id == idlist))] = 255

        ret, binary = cv2.threshold(is_comet_id_temp, 127, 255, 0)
        contour_info = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        _, contours, hierarchy = contour_info
        return len(contours)

    def get_roi(self, x, y, k, sec_pred):
        h, w = sec_pred.shape
        i_start = 0 if x < k else x - k
        i_end = h if x > h - k - 1 else x + k + 1
        j_start = 0 if y < k else y - k
        j_end = w if y > w - k - 1 else y + k + 1
        sec_pred_roi = sec_pred[i_start:i_end, j_start:j_end]
        return sec_pred_roi

    def save_result_s(self, data_dir, sele_dir, image_file_list, pred_label, confidence):
        img_file = os.path.join(data_dir, os.path.basename(image_file_list)[:-3] + "jpg")
        copyfile = os.path.join(sele_dir, os.path.basename(image_file_list)[:-3] + "jpg")
        shutil.copy(img_file, copyfile)
        current_app.logger.info("sele save ok")

    def do(self, gpu_id, weights_dir, data_dir, sele_dir, task_queue, count_queue):
        model = ModelResNetRoad(gpu_id=gpu_id, weights_dir=weights_dir)

        while True:
            if task_queue.empty():
                time.sleep(3)

            time11 = time.time()

            _task = task_queue.get()
            if not isinstance(_task, Task):
                break
            if _task._exit_flag:
                break
            pred_label = None
            confidence = None
            image_file = _task._image_path
            try:
                pred_label, confidence = model.do(os.path.join(data_dir, image_file))
            except Exception as e:
                current_app.logger.error("recognition error:{}".format(repr(e)))

            if pred_label is not None and confidence is not None:
                self.get_idOI(data_dir, sele_dir, image_file, pred_label, confidence, idOI=[0, 1, 2, 4, 6])
                os.remove(image_file)
                cut_img_file = os.path.join(os.path.dirname(image_file), 'cut-' + os.path.basename(image_file))
                os.remove(cut_img_file)
                count_queue.put(1)
            time12 = time.time()
            current_app.logger.info("{} finish in {} s".format(_task._image_path, time12 - time11))
