"""
Django settings for train_platform project.

Generated by 'django-admin startproject' using Django 2.1.3.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.1/ref/settings/
"""
# -*-coding:utf-8-*-
import os
import datetime
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '2+z08b!3dy6z*wxya6eoelr8giil@j4aa4=xx39_g16m27ol@('

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

ALLOWED_HOSTS = ["*"]
# APPEND_SLASH=False
# Application definition

INSTALLED_APPS = [
    "rest_framework", "rest_framework.authtoken", 'corsheaders', 'django.contrib.admin', 'django.contrib.auth',
    'django.contrib.contenttypes', 'django.contrib.sessions', 'django.contrib.messages', 'django.contrib.staticfiles',
    "Apps.train_app.apps.AppsConfig", "Apps.data_app.apps.AppsConfig", "Apps.model_app.apps.AppsConfig",
    "Apps.user_app.apps.AppsConfig", "djcelery"
]

MIDDLEWARE = [
'utils.middle.RequestLogMiddleWare',
    "corsheaders.middleware.CorsMiddleware",
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    # 'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    # 'utils.middle.ValidTokenMiddleware',
]

ROOT_URLCONF = 'train_platform.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, 'templates')],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'train_platform.wsgi.application'

# Database
# https://docs.djangoproject.com/en/2.1/ref/settings/#databases

DATABASES = {
    # 'default': {
    #     'ENGINE': 'django.db.backends.mysql',
    #     'NAME': 'train',    #你的数据库名称
    #     'USER': 'root',    #你的数据库用户名
    #     'PASSWORD': '123456',    #你的数据库密码
    #     'HOST': '10.11.5.14',    #你的数据库主机，留空默认为localhost
    #     'PORT': '3306',    #你的数据库端口
    # },
    'default': {
        'ENGINE': 'django.db.backends.postgresql_psycopg2',
        'NAME': 'train_manage',    #你的数据库名称
        'USER': 'train_manage',    #你的数据库用户名
        'PASSWORD': 'train_manage',    #你的数据库密码
        'HOST': '192.168.5.34',    #你的数据库主机，留空默认为localhost
        'PORT': '5432',    #你的数据库端口
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.BasicAuthentication',
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework_jwt.authentication.JSONWebTokenAuthentication',
    )
}

JWT_AUTH = {
    'JWT_EXPIRATION_DELTA': datetime.timedelta(seconds=60*60*60),
    'JWT_RESPONSE_PAYLOAD_HANDLER': 'Apps.user_app.utils.jwt_response_payload_handler',
    'JWT_AUTH_HEADER_PREFIX': 'JWT',
    'JWT_GET_USER_SECRET_KEY': 'Apps.user_app.utils.jwt_get_secret_key',
}
# Internationalization
# https://docs.djangoproject.com/en/2.1/topics/i18n/

AUTHENTICATION_BACKENDS = [
    'Apps.user_app.utils.CustomBackend',
]

LANGUAGE_CODE = 'zh-hans'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True
STATIC_URL = '/static/'
STATICFILES_DIRS = [os.path.join(BASE_DIR, "static")]
# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.1/howto/static-files/
# 允许所有域名跨域(优先选择)
CORS_ORIGIN_ALLOW_ALL = True
CORS_ALLOW_CREDENTIALS = True

AUTH_USER_MODEL = "user_app.UserInfo"

DOCKER_URL = "kd-bd02.kuandeng.com/kd-recog/"

CACHES = {
    'default': {
        'BACKEND': 'django.core.cache.backends.memcached.MemcachedCache',
        'LOCATION': 'unix:/tmp/memcached.sock',
        'KEY_PREFIX': 'lcfcn',
        'TIMEOUT': None
    }
}

import time

cur_path = os.path.dirname(os.path.realpath(__file__))    # log_path是存放日志的路径
log_path = os.path.join(os.path.dirname(cur_path), 'logs')
if not os.path.exists(log_path): os.mkdir(log_path)    # 如果不存在这个logs文件夹，就自动创建一个

LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
    # 日志格式
        'standard': {
            'format': '[%(asctime)s] [%(filename)s:%(lineno)d] [%(module)s:%(funcName)s] '
            '[%(levelname)s]- %(message)s'
        },
        'simple': {    # 简单格式
            'format': '%(levelname)s %(message)s'
        },
    },
    # 过滤
    'filters': {},
    # 定义具体处理日志的方式
    'handlers': {
    # 默认记录所有日志
        'default': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_path, 'all-{}.log'.format(time.strftime('%Y-%m-%d'))),
            'maxBytes': 1024 * 1024 * 5,    # 文件大小
            'backupCount': 5,    # 备份数
            'formatter': 'standard',    # 输出格式
            'encoding': 'utf-8',    # 设置默认编码，否则打印出来汉字乱码
        },
    # 输出错误日志
        'error': {
            'level': 'ERROR',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_path, 'error-{}.log'.format(time.strftime('%Y-%m-%d'))),
            'maxBytes': 1024 * 1024 * 5,    # 文件大小
            'backupCount': 5,    # 备份数
            'formatter': 'standard',    # 输出格式
            'encoding': 'utf-8',    # 设置默认编码
        },
    # 控制台输出
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
    # 输出info日志
        'info': {
            'level': 'INFO',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(log_path, 'info-{}.log'.format(time.strftime('%Y-%m-%d'))),
            'maxBytes': 1024 * 1024 * 5,
            'backupCount': 5,
            'formatter': 'standard',
            'encoding': 'utf-8',    # 设置默认编码
        },
    },
    # 配置用哪几种 handlers 来处理日志
    'loggers': {
    # 类型 为 django 处理所有类型的日志， 默认调用
        'django': {
            'handlers': ['default',"error", 'console'],
            'level': 'INFO',
            'propagate': False
        },
    # log 调用时需要当作参数传入
        'log': {
            'handlers': ['error', 'info', 'console', 'default'],
            'level': 'INFO',
            'propagate': True
        },
    }
}

import djcelery
djcelery.setup_loader()

BROKER_URL = 'redis://10.11.5.165:6379/3'

# CELERY_RESULT_BACKEND = 'redis://10.11.5.165:6379/1'
CELERY_RESULT_BACKEND = 'djcelery.backends.database:DatabaseBackend'
CELERY_IMPORTS = (
    'Apps.data_app.tasks',
    'Apps.train_app.tasks',
    'Apps.model_app.tasks',
)
CELERY_TIMEZONE = TIME_ZONE
CELERYBEAT_SCHEDULER = 'djcelery.schedulers.DatabaseScheduler'

from celery.schedules import crontab
CELERYBEAT_SCHEDULE = {
    #定时任务一：　每24小时周期执行任务(del_redis_data)
    u'删除过期的数据': {
        "task": "Apps.data_app.tasks.clear_data",
        "schedule": crontab(hour='*/24'),
        "args": (),
    },
}

CELERY_RESULT_SERIALIZER = 'json'
DATA_UPLOAD_MAX_NUMBER_FIELDS = 1024000



EMAIL_BACKEND = 'django.core.mail.backends.smtp.EmailBackend'
EMAIL_HOST = 'smtp.partner.outlook.cn'
EMAIL_PORT = 587
EMAIL_USE_TLS = True
# 发送邮件的邮箱
EMAIL_HOST_USER = 'liuhui@kuandeng.com'
# 在邮箱中设置的客户端授权密码
EMAIL_HOST_PASSWORD = 'Kdtemp02'
# 收件人看到的发件人
EMAIL_FROM = '自动报告<train_platform@kuandeng.com>'
