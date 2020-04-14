# Generated by Django 2.1.3 on 2020-02-18 02:27

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        ('train_app', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='InferTestTask',
            fields=[
                ('id', models.AutoField(db_column='id', primary_key=True, serialize=False, verbose_name='ID')),
                ('task_id', models.CharField(db_column='task_id', max_length=255, verbose_name='任务ID')),
                ('start_time', models.DateTimeField(db_column='start_time', max_length=255, verbose_name='开始时间')),
                ('end_time', models.DateTimeField(db_column='end_time', max_length=255, verbose_name='结束时间')),
                ('host_id', models.CharField(db_column='host_id', max_length=255, verbose_name='服务器')),
                ('gpu_id', models.CharField(db_column='gpu_id', max_length=255, verbose_name='GPU')),
                ('types', models.CharField(db_column='types', max_length=255, verbose_name='类型')),
                ('data_name', models.CharField(db_column='data_name', max_length=255, verbose_name='数据目录')),
                ('infos', models.TextField(blank=True, db_column='infos', max_length=4000, verbose_name='备注')),
                ('status', models.CharField(db_column='status', max_length=255, verbose_name='状态')),
                ('infer_task_type', models.ForeignKey(db_column='infer_task_type', on_delete=django.db.models.deletion.CASCADE, related_name='infer_task_type', to='train_app.TrainTask', verbose_name='任务类型')),
                ('infer_version', models.ForeignKey(db_column='infer_version', on_delete=django.db.models.deletion.CASCADE, related_name='infer_version', to='train_app.TrainTask', verbose_name='服务版本')),
            ],
            options={
                'verbose_name': '推断验证任务',
                'verbose_name_plural': '推断验证任务',
                'db_table': 'infer_test_task',
            },
        ),
        migrations.CreateModel(
            name='ModelRecord',
            fields=[
                ('id', models.AutoField(db_column='id', primary_key=True, serialize=False, verbose_name='ID')),
                ('task_id', models.CharField(db_column='task_id', max_length=255, verbose_name='任务ID')),
                ('time_info', models.DateTimeField(db_column='time_info', max_length=255, verbose_name='开始时间')),
                ('task_class', models.TextField(db_column='task_class', max_length=255, verbose_name='任务类别')),
                ('task_data', models.CharField(db_column='task_data', max_length=255, verbose_name='任务数据')),
                ('use_model', models.CharField(db_column='use_model', max_length=255, verbose_name='模型')),
                ('status', models.CharField(db_column='status', max_length=255, verbose_name='状态')),
                ('task_types', models.ForeignKey(db_column='task_types', on_delete=django.db.models.deletion.CASCADE, related_name='task_types', to='train_app.TrainTask', verbose_name='任务类型')),
            ],
            options={
                'verbose_name': '模型记录',
                'verbose_name_plural': '模型记录',
                'db_table': 'model_record',
            },
        ),
        migrations.CreateModel(
            name='ReleaseModel',
            fields=[
                ('id', models.AutoField(db_column='id', primary_key=True, serialize=False, verbose_name='ID')),
                ('task_id', models.CharField(db_column='task_id', max_length=255, verbose_name='任务ID')),
                ('release_time', models.DateTimeField(db_column='release_time', max_length=255, verbose_name='开始时间')),
                ('infos', models.TextField(blank=True, db_column='infos', max_length=4000, verbose_name='备注')),
                ('status', models.CharField(db_column='status', max_length=255, verbose_name='状态')),
                ('release_task_name', models.ForeignKey(db_column='release_task_name', on_delete=django.db.models.deletion.CASCADE, to='train_app.TrainTask', verbose_name='任务名')),
                ('release_task_type', models.ForeignKey(db_column='release_task_type', on_delete=django.db.models.deletion.CASCADE, related_name='release_task_type', to='train_app.TrainTask', verbose_name='任务类型')),
                ('release_version', models.ForeignKey(db_column='release_version', on_delete=django.db.models.deletion.CASCADE, related_name='release_version', to='train_app.TrainTask', verbose_name='服务版本')),
            ],
            options={
                'verbose_name': '发布模型',
                'verbose_name_plural': '发布模型',
                'db_table': 'release_model',
            },
        ),
    ]