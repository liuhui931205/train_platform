# coding=utf-8

from flask import Blueprint

api = Blueprint('api_1_0', __name__)

from . import auto_datas, training_task, network, datas, label_map, evaluate, model, release, log_file, gpu_info, offline_import, line_download, task_divide, pro_label, tensorRT_wotk, near_farm, label_data, score_data
