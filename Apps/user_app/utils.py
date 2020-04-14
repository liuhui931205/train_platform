# from .serializer import CreateUserSerializer
from utils.response_code import RET
from django.contrib.auth.backends import ModelBackend
from rest_framework_jwt.serializers import JSONWebTokenSerializer
from rest_framework.serializers import ValidationError
from rest_framework import serializers
from .models import UserInfo
from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer


def jwt_response_payload_handler(token, user=None, request=None):
    """
    自定义jwt认证成功返回数据
    """
    return {"errno": RET.OK, "data": {'token': token, 'username': user.username}, "message": "success"}


class CustomBackend(ModelBackend):
    """
    用户自定义用户验证
    """

    def authenticate(self, request, username=None, password=None, **kwargs):    # 重写这个函数
        try:
            # user = JSONWebTokenSerializer().validate(request)
            user = UserInfo.objects.get(username=username)
            if user.check_password(password):    #加密后比对前端密码
                return user
        except Exception as e:
            return None

        else:
            return None


def get_username(token):
    valid_data = VerifyJSONWebTokenSerializer().validate(token)
    user = valid_data['user']
    return user


def jwt_get_secret_key(user_model):
    return user_model.user_jwt
