# -*- coding:utf-8 -*-
from flask import Flask
from config import config_dict
import config
import redis
from Apps.utils.commons import RegexConverter
from flask_sqlalchemy import SQLAlchemy
from flask_compress import Compress
import logging
from logging.handlers import RotatingFileHandler
from flask_cors import *
from celery import Celery
import json

db = SQLAlchemy()
async_mode = "threading"
redis_store = None
celery = Celery(__name__, broker=config.CELERY_BROKER_URL, backend=config.CELERY_RESULT_BACKEND)
mach_id = None
krs = None
kms = None
kds_meta = None
mark_get = None


def set_logging(log_level):
    # 设置日志的记录等级
    logging.basicConfig(level=log_level)    # 调试debug级
    # 创建日志记录器，指明日志保存的路径、每个日志文件的最大大小、保存的日志文件个数上限
    file_log_handler = RotatingFileHandler("logs/log", maxBytes=1024 * 1024 * 100, backupCount=100000)
    # 创建日志记录的格式                 日志等级    输入日志信息的文件名 行数    日志信息
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(filename)s:%(lineno)d %(message)s')
    # 为刚创建的日志记录器设置日志记录格式
    file_log_handler.setFormatter(formatter)
    # 为全局的日志工具对象（flask app使用的）添加日志记录器
    logging.getLogger().addHandler(file_log_handler)


def create_app(config_name):
    # 创建Flask应用程序实例
    app = Flask(__name__)
    CORS(app, supports_credentials=True)
    # 获取对应的配置类
    config_cls = config_dict[config_name]
    global mach_id, krs, kms, kds_meta, mark_get
    con_path = config_cls.con_path
    with open(con_path, 'r') as f:
        data = json.loads(f.read())
        mach_id=data["mach_id"]
        krs = data["krs"]
        kms = data["kms"]
        mark_get = data["mark_get"]
        kds_meta = data["kds_meta"]
    # 加载配置
    app.config.from_object(config_cls)
    set_logging(config_cls.LOG_LEVEL)
    db.init_app(app)
    Compress(app)

    global redis_store
    redis_store = redis.StrictRedis(host=config_cls.REDIS_HOST, port=config_cls.REDIS_PORT, db=config_cls.REDIST_DB)

    app.url_map.converters['re'] = RegexConverter
    celery.conf.update(app.config)
    # 注册蓝图
    from Apps.api import api
    app.register_blueprint(api, url_prefix="/api")
    # from Apps.web_html import html
    # app.register_blueprint(html)

    return app
