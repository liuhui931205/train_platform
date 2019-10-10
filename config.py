# -*- coding:utf-8 -*-
import logging
import multiprocessing
import os



class Config(object):
    SECRET_KEY = 'qyEzGidVnaRZNInFA6lO7AoPgIJGr83Em+wXttn8rBEGnbRswiviq5moyKDXG21j'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    CLIENT_MULTI_STATEMENTS = True
    JSONIFY_PRETTYPRINT_REGULAR = False
    SQLALCHEMY_COMMIT_ON_TEARDOWN = True

    REDIS_HOST = '10.11.5.165'
    REDIS_PORT = 6379
    REDIST_DB = 4


class DevelopmentConfig(Config):
    """开发阶段的配置类"""
    DEBUG = True
    LOG_LEVEL = logging.DEBUG
    # SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@10.11.5.165:3306/trains'
    SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@10.11.5.14:3306/trains'
    COMPRESS_MIMETYPES = ['application/json']
    COMPRESS_MIN_SIZE = 500
    con_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.cfg")


class ProductionConfig(Config):
    """生产阶段的配置类"""
    # 数据库配置
    LOG_LEVEL = logging.INFO
    # SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@10.11.5.165:3306/trains'
    SQLALCHEMY_DATABASE_URI = 'mysql://root:123456@10.11.5.14:3306/trains'
    COMPRESS_MIMETYPES = ['application/json']
    COMPRESS_MIN_SIZE = 500
    con_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),"config.cfg")



# 配置
config_dict = {'development': DevelopmentConfig, 'production': ProductionConfig}

# 映射
url_mach = {
    "161": "http://10.11.5.161:9001/api/",
    "162": "http://10.11.5.162:9001/api/",
    "163": "http://10.11.5.163:9001/api/",
    "164": "http://10.11.5.164:9001/api/",
    "165": "http://10.11.5.165:9001/api/",
}

# 并行训练机器
parallel_training = [
    {
    "dest_scp_ip": "10.11.5.161",
    "dest_scp_port": 22,
    "dest_scp_user": "kdreg",
    "dest_scp_passwd": "kd-123",
    },
    {
    "dest_scp_ip": "10.11.5.162",
    "dest_scp_port": 22,
    "dest_scp_user": "kdreg",
    "dest_scp_passwd": "kd-123",
}, {
    "dest_scp_ip": "10.11.5.164",
    "dest_scp_port": 22,
    "dest_scp_user": "kdreg",
    "dest_scp_passwd": "kd-123",
}, {
    "dest_scp_ip": "10.11.5.165",
    "dest_scp_port": 22,
    "dest_scp_user": "kdreg",
    "dest_scp_passwd": "kd-123",
}
#     , {
#     "dest_scp_ip": "192.168.7.20",
#     "dest_scp_port": 22,
#     "dest_scp_user": "shencheng",
#     "dest_scp_passwd": "kd-123",
# }
]
parallel_map = {
    "161": {
        "dest_scp_ip": "10.11.5.161",
        "dest_scp_port": 22,
        "dest_scp_user": "kdreg",
        "dest_scp_passwd": "kd-123",
    },
    "162": {
        "dest_scp_ip": "10.11.5.162",
        "dest_scp_port": 22,
        "dest_scp_user": "kdreg",
        "dest_scp_passwd": "kd-123",
    },
    "164": {
        "dest_scp_ip": "10.11.5.164",
        "dest_scp_port": 22,
        "dest_scp_user": "kdreg",
        "dest_scp_passwd": "kd-123",
    },
    "165": {
        "dest_scp_ip": "10.11.5.165",
        "dest_scp_port": 22,
        "dest_scp_user": "kdreg",
        "dest_scp_passwd": "kd-123",
    }
    # ,
    # "20": {
    #     "dest_scp_ip": "192.168.7.20",
    #     "dest_scp_port": 22,
    #     "dest_scp_user": "shencheng",
    #     "dest_scp_passwd": "kd-123",
    # }

}

# ssh地址
ssh_dress11 = {
    "dest_scp_ip": "10.11.5.161",
    "dest_scp_port": 76,
    "dest_scp_user": "root",
    "dest_scp_passwd": "1234567890",
}

ssh_dress12 = {
    "dest_scp_ip": "10.11.5.162",
    "dest_scp_port": 76,
    "dest_scp_user": "root",
    "dest_scp_passwd": "1234567890",
}

ssh_dress13 = {
    "dest_scp_ip": "10.11.5.163",
    "dest_scp_port": 76,
    "dest_scp_user": "root",
    "dest_scp_passwd": "1234567890",
}

ssh_dress14 = {
    "dest_scp_ip": "10.11.5.164",
    "dest_scp_port": 76,
    "dest_scp_user": "root",
    "dest_scp_passwd": "1234567890",
}

ssh_dress15 = {
    "dest_scp_ip": "10.11.5.165",
    "dest_scp_port": 76,
    "dest_scp_user": "root",
    "dest_scp_passwd": "12345678",
}


ssh_dress00 = {
    "dest_scp_ip": "10.11.5.201",
    "dest_scp_port": 77,
    "dest_scp_user": "root",
    "dest_scp_passwd": "12345678",
}

data0_sour_data = "/data/deeplearning/dataset/kd/lane"
data0_dest_data = "/data/deeplearning/liuhui/train_platform/data"
data0_sour_train = "/data/deeplearning/liuhui/train_platform/train_task"
data_sour_data = "/data/deeplearning/dataset/kd/lane"
data_dest_data = "/data/deeplearning/train_platform/data"
data_sour_train = "/data/deeplearning/train_platform/train_task"

json_path = "/data/deeplearning/train_platform/json_dirs"
class_id = [0, 1]


# 打包和发布
class GlobalVar(object):
    filter_data = list()
    manager = multiprocessing.Manager()
    image_dir = manager.Value("s", "/data/deeplearning/dataset/training/data/images")
    package_dir = manager.Value("s", "/data/deeplearning/dataset/training/data/packages")
    check_dir = manager.Value("s", "/data/deeplearning/dataset/training/data/released")
    release_dir = manager.Value("s", "/data/deeplearning/dataset/kd/lane")
    krs_url = manager.Value("s", "http://192.168.5.69:23100/krs")
    release_url = manager.Value("s", "http://192.168.5.69:23300/kts")

    model_host = manager.Value("s", "192.168.7.100")
    model_user = manager.Value("s", "root")
    model_port = manager.Value("i", 76)
    model_passwd = manager.Value("s", "12345678")


# 任务类型
task_type = ['classification-model', 'target-detection', 'semantic-segmentation', 'OCR']

CELERY_BROKER_URL = "redis://10.11.5.165:6379/2"
CELERY_RESULT_BACKEND = "redis://10.11.5.165:6379/3"
CELERYD_MAX_TASKS_PER_CHILD = 20
CELERY_TIMEZONE = 'Asia/Shanghai'
CELERYD_FORCE_EXECV = True
