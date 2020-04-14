# -*-coding:utf-8-*-
from django.db import models
from Apps.data_app.models import DataProHandle


class TrainTask(models.Model):
    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    start_time = models.DateTimeField("开始时间", db_column="start_time", max_length=255)
    end_time = models.DateTimeField("结束时间", db_column="end_time", max_length=255, null=True)
    host_id = models.CharField("服务器", db_column="host_id", max_length=255)
    gpu_id = models.CharField("GPU", db_column="gpu_id", max_length=255)
    task_name = models.CharField("任务名", db_column="task_name", max_length=255)
    train_data_name = models.ForeignKey(DataProHandle,
                                        on_delete=models.CASCADE,
                                        related_name="train_data_name",
                                        db_column="train_data_name",
                                        verbose_name='数据目录')
    task_type = models.CharField("任务类型", db_column="task_type", max_length=255)
    weight = models.CharField("初始模型", db_column="weight", max_length=255, null=True)
    infos = models.TextField("备注", db_column="infos", max_length=4000, null=True)
    version = models.CharField("服务版本", db_column="version", max_length=255, null=True)
    pro_id = models.CharField("进程ID", db_column="pro_id", max_length=255, null=True)
    status = models.CharField("状态", db_column="status", max_length=255, null=True)
    types = models.CharField("类型", db_column="types", max_length=255, null=True)
    sub_type = models.CharField("图片类型", db_column="sub_type", max_length=255, null=True)
    scene = models.CharField("场景", db_column="scene", max_length=255, null=True)
    adcode = models.CharField("地点", db_column="adcode", max_length=255, null=True)

    biz = models.CharField("业务", db_column="biz", max_length=255, null=True)
    customer = models.CharField("客户", db_column="customer", max_length=255, null=True)


    class Meta:
        db_table = "train_task"
        verbose_name = "训练任务"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.task_name

    def to_dict(self):

        select_dict = {
            'task_id': self.task_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'host_id': self.host_id,
            'gpu_id': self.gpu_id,
            'task_name': self.task_name,
            'data_name': self.train_data_name.dir_name,
            'infos': self.infos,
            'task_type': self.task_type,
            'weight': self.weight,
            'version': self.version,
            'types': self.types,
            'sub_type': self.sub_type,
            'scene': self.scene,
            'adcode': self.adcode,
            'biz': self.biz,
            'pro_id': self.pro_id,
            'customer': self.customer,
            "status": self.status
        }
        return select_dict
