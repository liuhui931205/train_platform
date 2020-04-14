# -*-coding:utf-8-*-
from django.db import models


class ProData(models.Model):
    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    pro_data_type = models.CharField("类型", db_column="pro_data_type", max_length=255)
    pro_dir_name = models.CharField("目录名", db_column="pro_dir_name", max_length=255)
    infos = models.TextField("备注", db_column="infos", max_length=4000, null=True)
    status = models.CharField("状态", db_column="status", max_length=255)

    class Meta:
        db_table = "pro_data"
        verbose_name = "预处理数据"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.pro_dir_name

    def to_dict(self):

        select_dict = {
            'task_id': self.task_id,
            'data_type': self.pro_data_type,
            'dir_name': self.pro_dir_name,
            'infos': self.infos,
            "status": self.status
        }
        return select_dict


class DataProHandle(models.Model):
    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    start_time = models.DateTimeField("开始时间", db_column="start_time", max_length=255)
    end_time = models.DateTimeField("结束时间", db_column="end_time", max_length=255, null=True)
    dir_name = models.CharField("目录名", db_column="dir_name", max_length=255)
    data_task_type = models.CharField("任务类型", db_column="data_task_type", max_length=255)
    data_version = models.CharField("服务版本", db_column="data_version", max_length=255)
    src_dir_name = models.CharField("源数据目录", db_column="src_dir_name", max_length=255)
    data_host_id = models.CharField("服务器", db_column="data_host_id", max_length=255, null=True)
    infos = models.TextField("备注", db_column="infos", max_length=4000, null=True)
    data_conf_path = models.CharField("配置文件", db_column="data_conf_path", max_length=512, null=True)
    status = models.CharField("状态", db_column="status", max_length=255)

    class Meta:
        db_table = "data_preprocess"
        verbose_name = "数据预处理"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.dir_name

    def to_dict(self):
        select_dict = {
            'task_id': self.task_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'dir_name': self.dir_name,
            'src_dir_name': self.src_dir_name,
            'infos': self.infos,
            "task_type": self.data_task_type,
            "version": self.data_version,
            "host": self.data_host_id,
            "conf_path": self.data_conf_path,
            "status": self.status
        }
        return select_dict
