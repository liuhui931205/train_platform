from rest_framework_jwt.settings import api_settings
from utils.response_code import RET
from rest_framework.exceptions import AuthenticationFailed
from rest_framework_jwt.serializers import VerifyJSONWebTokenSerializer
from rest_framework.serializers import ValidationError



class TokenAuth():

    def authenticate(self, request):
        token = {"token": None}
        # print(request.META.get("HTTP_TOKEN"))
        token["token"] = request.META.get('HTTP_TOKEN')
        try:
            valid_data = VerifyJSONWebTokenSerializer().validate(token)
            user = valid_data['user']
        except Exception as e:
            raise ValidationError({"errno": RET.USERERR, "data": None, "message": "Signature has expired"})
        else:
            if user:
                return
            else:
                raise AuthenticationFailed('认证失败')
