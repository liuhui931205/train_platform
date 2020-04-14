# -*-coding:utf-8-*-
from django.db import models
from Apps.train_app.models import TrainTask


class InferTestTask(models.Model):
    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    start_time = models.DateTimeField("开始时间", db_column="start_time", max_length=255)
    end_time = models.DateTimeField("结束时间", db_column="end_time", max_length=255, null=True)
    host_id = models.CharField("服务器", db_column="host_id", max_length=255)
    gpu_id = models.CharField("GPU", db_column="gpu_id", max_length=255)
    data_name = models.CharField("数据目录", db_column="data_name", max_length=255)
    use_model = models.CharField("模型", db_column="use_model", max_length=1000, null=True)
    infer_task_type = models.CharField("任务类型", db_column="infer_task_type", max_length=255)
    types = models.CharField("类型", db_column="types", max_length=255)
    seg_tag = models.CharField("识别服务版本", db_column="seg_tag", max_length=255, null=True)
    result = models.CharField("结果", db_column="result", max_length=255, null=True)
    infer_version = models.CharField("服务版本", db_column="infer_version", max_length=255)
    image_type = models.CharField("图片类型", db_column="image_type", max_length=255, null=True)
    status = models.CharField("状态", db_column="status", max_length=255)

    class Meta:
        db_table = "infer_test_task"
        verbose_name = "推断验证任务"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.task_id

    def to_dict(self):

        select_dict = {
            'task_id': self.task_id,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'host_id': self.host_id,
            'gpu_id': self.gpu_id,
            'types': self.types,
            'data_name': self.data_name,
            'infos': self.infos,
            'task_type': self.infer_task_type.task_type.data_task_type,
            'version': self.infer_version.version.data_version,
            "status": self.status
        }
        return select_dict


class ReleaseModel(models.Model):
    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    types = models.CharField("类型", db_column="types", max_length=255, null=True)
    sub_type = models.CharField("图片类型", db_column="sub_type", max_length=255, null=True)
    scene = models.CharField("场景", db_column="scene", max_length=255, null=True)
    adcode = models.CharField("地点", db_column="adcode", max_length=255, null=True)
    biz = models.CharField("业务", db_column="biz", max_length=255, null=True)
    customer = models.CharField("客户", db_column="customer", max_length=255, null=True)
    release_time = models.DateTimeField("发布时间", db_column="release_time", max_length=255)
    name = models.CharField("模型名称", db_column="name", max_length=255, null=True)
    author = models.CharField("作者", db_column="author", max_length=255, null=True)
    file_path = models.CharField("路径", db_column="file_path", max_length=512, null=True)
    description = models.TextField("备注", db_column="description", max_length=4000, null=True)
    model = models.CharField("模型", db_column="model", max_length=512, null=True)
    status = models.CharField("状态", db_column="status", max_length=255, null=True)

    class Meta:
        db_table = "release_model"
        verbose_name = "发布模型"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.task_id

    def to_dict(self):

        select_dict = {
            'task_id': self.task_id,
            'release_time': self.release_time,
            'task_name': self.release_task_name.task_name,
            'infos': self.infos,
            'task_type': self.release_task_type.task_type.data_task_type,
            'version': self.release_version.version.data_version,
            "status": self.status
        }
        return select_dict


class ModelRecord(models.Model):

    id = models.AutoField("ID", db_column="id", primary_key=True)
    task_id = models.CharField("任务ID", db_column="task_id", max_length=255)
    time_info = models.DateTimeField("开始时间", db_column="time_info", max_length=255)
    task_types = models.ForeignKey(TrainTask, db_column="task_types", on_delete=models.CASCADE,
                                          related_name="task_types", verbose_name='任务类型')
    task_class = models.TextField("任务类别", db_column="task_class", max_length=255)
    task_data = models.CharField("任务数据", db_column="task_data", max_length=255)
    use_model = models.CharField("模型", db_column="use_model", max_length=255)
    status = models.CharField("状态", db_column="status", max_length=255)

    class Meta:
        db_table = "model_record"
        verbose_name = "模型记录"
        verbose_name_plural = verbose_name

    def __str__(self):
        return self.task_id

    def to_dict(self):
        select_dict = {
            'task_id': self.task_id,
            'time_info': self.time_info,
            'task_types': self.task_types.task_type.data_task_type,
            'task_class': self.task_class,
            'task_data': self.task_data,
            'use_model': self.use_model,
            "status": self.status
        }
        return select_dict
