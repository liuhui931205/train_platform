# -*-coding:utf-8-*-
# from . import make_celery
# celery = make_celery()
import time
import numpy as np
import cv2
import json
from Apps import celery, db
from Apps.utils.utils import self_full_labels
import requests
from sqlalchemy import Column, DateTime, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from celery.utils.log import get_task_logger

Base = declarative_base()
metadata = Base.metadata
log = get_task_logger(__name__)


class Schedulefrom(Base):
    __tablename__ = 'progress'
    id = Column(Integer, primary_key=True)
    id_code = Column(String(255))
    task_id = Column(String(255))
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    picid = Column(String(255))
    statue = Column(String(255))


class LabelData(Base):
    __tablename__ = 'data_manage'
    id = Column(Integer, primary_key=True)
    time_info = Column(String(255))
    trackpointid = Column(String(255))
    imgrange = Column(String(255))
    city = Column(String(255))
    label_info = Column(String(255))
    tag_info = Column(String(4000))
    pacid = Column(String(255))


# class DownloadTask(object):
#
#     def __init__(self,
#                  track_point_id=None,
#                  pic_id=None,
#                  imgrange=None,
#                  city=None,
#                  label_info=None,
#                  time_info=None,
#                  exit_flag=False):
#         self.track_point_id = track_point_id
#         self.pic_id = pic_id
#         self.imgrange = imgrange
#         self.city = city
#         self.label_info = label_info
#         self.time_info = time_info
#         self.exit_flag = exit_flag


@celery.task(bind=True)
def store_data(self, picid, id_code):
    # for pic_id in pic_list:
    #     print pic_id
    succ_count = 0
    fail_count = 0
    exist_track = []
    try:
        traintask = db.session.query(LabelData).all()
        for task in traintask:
            exist_track.append(task.trackpointid)
        data_task = db.session.query(Schedulefrom).filter_by(id_code=id_code).first()
        data_task.statue = "progress"
        db.session.add(data_task)
        db.session.commit()

        url = "http://10.11.5.74:13320/kms/task/getMarkTagResult?pacId={}".format(picid)
        info = requests.get(url)
        json_data = json.loads(info.text)
        if json_data["code"] == "0":
            result = json_data["result"]["node"]
            total_count = len(result)
            for dt in result:
                time_info = dt["timestamp"]
                imgrange = "0"
                label_info = "1.0"
                city = ""
                pacid = ""
                trackpointid = ""
                st = time.localtime(int(time_info) / 1000.0)
                time_info = time.strftime('%Y-%m-%d %H:%M:%S', st)
                for dt_if in dt["tag"]:
                    if dt_if["k"] == "TRACKPOINTID":
                        trackpointid = dt_if["v"]
                    # elif dt_if["k"] == "handle":
                    #     handle = dt_if["v"]
                    elif dt_if["k"] == "IMGOPRANGE":
                        imgrange = dt_if["v"]
                    elif dt_if["k"] == "CITY":
                        city = dt_if["v"]
                    elif dt_if["k"] == "ROADELEMENT":
                        label_info = dt_if["v"]
                    elif dt_if["k"] == "PACID":
                        pacid = dt_if["v"]
                if not pacid:
                    pacid = picid
                url2 = "http://10.11.5.77:13100/krs/image/get?trackPointId={}&type=70&seq=001&imageType=png".format(
                    trackpointid)
                try:
                    img = requests.get(url2)
                    img_data = img.content
                    _image0 = np.asarray(bytearray(img_data), dtype="uint8")
                    origin_image = cv2.imdecode(_image0, cv2.IMREAD_COLOR)

                    width = origin_image.shape[1]
                    height = origin_image.shape[0]
                    tag_info = {}
                    for label in self_full_labels:
                        color = (label.color[2], label.color[1], label.color[0])
                        if color != (32, 64, 64) and color != (0, 0, 0) and color != (192, 192, 192):
                            c = np.where((origin_image == color).all(axis=2))
                            if len(c[0]) != 0:
                                tag_info[label.categoryId] = {
                                    "px_count": str(len(c[0])),
                                    "px_prop": str(round(len(c[0]) / (2048 * 2448 * 0.01), 4)) + "%",
                                    "name": label.name
                                }

                    if trackpointid in exist_track:
                        data = db.session.query(LabelData).filter_by(trackpointid=trackpointid).first()
                        data.time_info = time_info
                        data.imgrange = imgrange
                        data.pacid = pacid
                        data.city = city
                        data.label_info = label_info
                        data.tag_info = json.dumps(tag_info)
                    else:
                        data = LabelData()
                        data.time_info = time_info
                        data.trackpointid = trackpointid
                        data.imgrange = imgrange
                        data.pacid = pacid
                        data.city = city
                        data.label_info = label_info
                        data.tag_info = json.dumps(tag_info)

                    db.session.add(data)
                    db.session.commit()
                except Exception as e:
                    log.info(trackpointid)
                    log.error(e)
                    fail_count += 1
                else:
                    succ_count += 1
                finally:
                    if (succ_count + fail_count) != total_count:
                        self.update_state(
                            state='PROGRESS',
                            meta={
                                'success': succ_count,
                                'failed': fail_count,
                                'total': total_count,
                                'status': str(((succ_count + fail_count) * 1.0) / total_count * 100.0) + "%"
                            })
                    else:
                        data_task = db.session.query(Schedulefrom).filter_by(id_code=id_code).first()
                        data_task.statue = "completed!,success:{},failed:{}".format(succ_count, fail_count)
                        data_task.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
                        db.session.add(data_task)
                        db.session.commit()
    except Exception as e:
        log.error(e)
        data_task = db.session.query(Schedulefrom).filter_by(id_code=id_code).first()
        data_task.statue = "failure"
        data_task.end_time = time.strftime("%Y-%m-%d %H:%M", time.localtime())
        db.session.add(data_task)
        db.session.commit()
