# coding=utf-8
from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'start_consistency_task$', views.modelConsistency.as_view(), name="start_consistency_task"),
    url(r'query_model$', views.modelQuery.as_view(), name="query_model"),
    url(r'build_model_test$', views.modelTest.as_view(), name="build_model_test"),
    url(r'query_model_conf$', views.modelTestConf.as_view(), name="query_model_conf"),
    url(r'save_model_conf$', views.modelTestConf.as_view(), name="save_model_conf"),
    url(r'start_model_task$', views.modelTest.as_view(), name="start_model_task"),
    url(r'standard$', views.visualization.as_view(), name="standard"),
    url(r'display$', views.visualization.as_view(), name="display"),
    url(r'download_model_info$', views.downloadImage.as_view(), name="download_model_info"),
    url(r'query_con_conf$', views.modelConsistencyConf.as_view(), name="query_con_conf"),
    url(r'save_con_conf$', views.modelConsistencyConf.as_view(), name="save_con_conf"),
    url(r'build_consistency$', views.modelConsistency.as_view(), name="build_consistency"),
    url(r'build_release$', views.modelRelease.as_view(), name="build_release"),
    url(r'start_release_task$', views.modelRelease.as_view(), name="start_release_task"),
    url(r'start_tensorrt_task$', views.modelToTensorRt.as_view(), name="start_tensorrt_task"),
    url(r'build_tensorrt$', views.modelToTensorRt.as_view(), name="build_tensorrt"),
    url(r'query_tensorrt_conf$', views.modelTensorRtConf.as_view(), name="query_tensorrt_conf"),
    url(r'save_tensorrt_conf$', views.modelTensorRtConf.as_view(), name="save_tensorrt_conf"),
    url(r'query_seg_tag$', views.querySegnet.as_view(), name="query_seg_tag"),
    url(r'online_model$', views.modelOnline.as_view(), name="online_model"),
]