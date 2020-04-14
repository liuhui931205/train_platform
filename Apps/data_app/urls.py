# coding=utf-8
from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'build_data_task$', views.dataProHandle.as_view(), name="build_data_task"),
    url(r'start_data_task$', views.dataProHandle.as_view(), name="start_data_task"),
    url(r'train_data$', views.dataQuery.as_view(), name="train_data"),
    url(r'query_data$', views.dataQuery.as_view(), name="query_data"),
    url(r'query_data_conf$', views.dataConf.as_view(), name="query_data_conf"),
    url(r'save_data_conf$', views.dataConf.as_view(), name="save_data_conf"),
    url(r'local_upload$', views.localUpload.as_view(), name="local_upload"),
    url(r'recog_conf$', views.recogConf.as_view(), name="recog_conf"),
    url(r'recog_class$', views.recogClass.as_view(), name="recog_class"),
    url(r'docker_ver$', views.DockerVer.as_view(), name="docker_ver"),
    url(r'src_query$', views.srcData.as_view(), name="src_query"),
]