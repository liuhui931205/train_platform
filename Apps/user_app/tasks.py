import os
from train_platform.celery import app
from django.core.mail import send_mail
from django.conf import settings
import logging
from billiard.exceptions import Terminated
import time
logger = logging.getLogger("django")


@app.task(bind=True, throws=(Terminated,))
def send_email(self, to_email):
# def send_email(to_email):
    subject = "自动报告"    # 标题
    body = "识别完成"    # 文本邮件体
    sender = settings.EMAIL_HOST_USER    # 发件人
    receiver = [to_email, "zhangmingming@kuandeng.com"]    # 接收人
    html_body = '<h1>尊敬的用户</h1>'
    send_mail(subject, body, sender, receiver, html_message=html_body)
