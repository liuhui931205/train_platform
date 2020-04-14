# -*-coding:utf-8-*-
from django.views import View
import json
from utils.response_code import RET
from django.http import JsonResponse
from rest_framework_jwt.utils import jwt_payload_handler, jwt_encode_handler
from django.contrib import auth
from Apps.user_app.utils import get_username
import logging
from Apps.user_app.models import UserInfo
from rest_framework_jwt.settings import api_settings
# from .serializer import CreateUserSerializer
from .tasks import send_email
from uuid import uuid4

logger = logging.getLogger("django")
# Create your views here.


class Register(View):
    # 内置注册普通 用户
    def get(self, request):
        pass

    def post(self, request):
        json_result = json.loads(request.body)
        username = json_result["username"]
        password = json_result["password"]
        try:
            user = UserInfo.objects.create_user(username=username, password=password)
            user.default_docker = "v1.0.0"
            user.save()
        except Exception as e:
            import traceback
            logger.error(traceback.format_exc())

            resp = JsonResponse({"errno": RET.DBERR, "data": None, "message": "failed"})
        else:
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
        return resp


class UserLog(View):

    def get(self, request):
        token = {"token": None}
        token["token"] = request.META.get('HTTP_TOKEN')
        try:
            user_name = get_username(token)
        except Exception as e:
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "用户已注销"})
        else:
            user = UserInfo.objects.get(username=user_name)
            user.user_jwt = uuid4()
            user.save()
            resp = JsonResponse({"errno": RET.OK, "data": None, "message": "注销成功"})
        return resp

    def post(self, request):
        json_result = json.loads(request.body)
        username = json_result["username"]
        password = json_result["password"]
        if not all([username, password]):
            resp = JsonResponse({"errno": RET.LOGINERR, "data": None, "message": "未填写用户名和密码"})
        else:
            user = auth.authenticate(username=username, password=password)

            if user is not None:
                payload = jwt_payload_handler(user)
                resp = JsonResponse({
                    "errno": RET.OK,
                    "data": {
                        'token': jwt_encode_handler(payload)
                    },
                    "message": "登录成功"
                })
                # 调用settings.py中JWT_AUTH的JWT_ENCODE_HANDLER，此处使用的是
                # 自带的处理函数，rest_framework_jwt.utils.jwt_encode_handler，生成token
            else:
                resp = JsonResponse({"errno": RET.USERERR, "data": None, "message": "用户名或密码不正确"})
        return resp


class sendMail(View):

    def get(self, request):
        pass

    def post(self, request):
        json_result = json.loads(request.body)
        email = json_result["email"]
        send_email.apply_async(args=[email], countdown=5)
        # send_email(email)

        resp = JsonResponse({"errno": RET.OK, "data": None, "message": "success"})
        return resp
