# coding=utf-8
from django.conf.urls import url
from . import views
# from rest_framework_jwt.views import obtain_jwt_token

urlpatterns = [
    url('user_register', views.Register.as_view(), name="user_register"),
    url(r'user_login', views.UserLog.as_view(), name="user_login"),
    url(r'user_logout', views.UserLog.as_view(), name="user_logout"),
    url(r'send_email', views.sendMail.as_view(), name="send_email"),
]
