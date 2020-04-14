# coding=utf-8
from django.conf.urls import url
from . import views


urlpatterns = [
    url(r'query_train_conf$', views.trainConf.as_view(), name="query_train_conf"),
    url(r'save_train_conf$', views.trainConf.as_view(), name="save_train_conf"),
    url(r'build_train_task$', views.trainBuild.as_view(), name="build_train_task"),
    url(r'query_init_model$', views.trainBuild.as_view(), name="query_init_model"),
    url(r'run_task$', views.trainStart.as_view(), name="run_task"),
    url(r'op_task$', views.trainStart.as_view(), name="op_task"),
    url(r'op_conf$', views.opConf.as_view(), name="op_conf"),
    url(r'op_log$', views.trainLog.as_view(), name="op_log"),
    url(r'query_log$', views.trainLog.as_view(), name="query_log"),
    url(r'train_history$', views.trainQuery.as_view(), name="train_history"),
    url('', views.Index.as_view(), name="index"),
]