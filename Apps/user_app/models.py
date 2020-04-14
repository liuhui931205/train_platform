# -*-coding:utf-8-*-
from django.db import models
from uuid import uuid4
from django.contrib.auth.models import User, AbstractUser


# 通过继承  扩展
class UserInfo(AbstractUser):
    default_docker = models.CharField(max_length=512, null=True)
    user_jwt = models.UUIDField(default=uuid4, verbose_name='用户jwt加密秘钥')

    class Meta:
        db_table = "user_info"
        verbose_name = "用户管理"
        verbose_name_plural = verbose_name
