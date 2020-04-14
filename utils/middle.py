from utils.response_code import RET
from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer
from rest_framework.serializers import ValidationError
from jwt import InvalidSignatureError
from django.http import HttpResponse
from uuid import uuid4
from django.utils.deprecation import MiddlewareMixin
import json

class RequestLogMiddleWare(MiddlewareMixin):

    # def __init__(self, get_response):
    #     self.get_response = get_response
    #
    # def __call__(self, request):
    #     response = self.get_response(request)
    #     return response

    def process_template_response(self, request, response):
        if hasattr(response, 'data'):
            if 'errno' in response.data:
                if type(response.data['errno']) is list:
                    response.data['errno'] = str(response.data['errno'][0])
                    response.data['message'] = str(response.data['message'][0])
                    response.data['data'] = None
            else:
                message = ""
                if isinstance(response.data, dict):
                    for k, v in response.data.items():
                        message = k + str(v[0])
                    # message = str(response.data)
                    response.data = {}
                    response.data['errno'] = RET.SERVERERR
                    response.data['message'] = message
                    response.data['data'] = None
                else:
                    for i in response.data:
                        message = str(i)
                    response.data = {}
                    response.data['errno'] = RET.SERVERERR
                    response.data['message'] = message
                    response.data['data'] = None
        return response

    def process_response(self, request, response):
        # 仅用于处理 login请求
        if request.META['PATH_INFO'] == '/api/user/user_login':
            if json.loads(response.content)["errno"] == "0":
                rep_data = json.loads(response.content)["data"]
                valid_data = VerifyJSONWebTokenSerializer().validate(rep_data)
                user = valid_data['user']
                user.save()
                return response
            else:
                return response
        else:
            return response


class ValidTokenMiddleware(object):

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        return response

    def process_request(self, request):
        # 用于处理 所有带 jwt 的请求
        jwt_token = request.META.get('HTTP_AUTHORIZATION', None)
        if jwt_token is not None and jwt_token != '':
            data = {
                'token': request.META['HTTP_AUTHORIZATION'].split(' ')[1],
            }
            try:
                valid_data = VerifyJSONWebTokenSerializer().validate(data)
                user = valid_data['user']
            except (InvalidSignatureError, ValidationError):
                # 找不到用户
                return HttpResponse("{'msg','请重新登入'}", content_type='application/json', status=400)
            if user.user_jwt != data['token']:
                user.user_jwt = uuid4()
                user.save()
                return HttpResponse("{'msg','请重新登入'}", content_type='application/json', status=400)

    def process_response(self, request, response):
        # 仅用于处理 login请求
        if request.META['PATH_INFO'] == '/api/user/user_login':
            rep_data = response.data
            valid_data = VerifyJSONWebTokenSerializer().validate(rep_data)
            user = valid_data['user']
            user.user_jwt = rep_data['token']
            user.save()
            return response
        else:
            return response
