# -*-coding:utf-8-*-
import os
from Apps import db
from Apps.utils.client import client
import shutil
from Apps.modules.base_data import BaseDataTask
from Apps.libs.unpack_dataset import start_unpack, start_unpack_d
from Apps.models import TrainTask, Datas
from config import data0_sour_data,data0_dest_data,data_sour_data,data_dest_data,data_sour_train,data0_sour_train
from Apps.utils.copy_all import copyFiles


def copy_data(l_value, dicts, mach_id, type):
    models = []
    if dicts:
        if l_value:
            lis = os.listdir('/data/deeplearning/train_platform/eva_sour_data/s_data/test_data')
            if len(lis) > 0:
                for j in lis:
                    os.remove(os.path.join('/data/deeplearning/train_platform/eva_sour_data/s_data/test_data', j))

        sour = '/data/deeplearning/train_platform/eva_sour_data/s_data'
        sour_li = []
        for k, v in dicts.items():
            if v:
                train_task = db.session.query(TrainTask).filter_by(task_name=k).first()
                di = {}
                if train_task:
                    task_type = train_task.type
                    machine_id = train_task.machine_id
                    data_desc_id = train_task.data_desc_id
                    datas = db.session.query(Datas).filter_by(id=data_desc_id).first()
                    data_name = datas.data_name
                    data_type = datas.data_type
                    get_path = os.path.join(data_sour_train, k, 'output/models')
                    sour_get_path = os.path.join(data0_sour_train, k, 'output/models')
                    map_path = os.path.join(data_sour_train, k, 'conf')
                    sour_map_path = os.path.join(data0_sour_train, k, 'conf')
                    if not os.path.exists(get_path):
                        os.makedirs(get_path)
                    if not os.path.exists(map_path):
                        os.makedirs(map_path)
                    # if machine_id != mach_id:
                    dest_scp, dest_sftp, dest_ssh = client(id="host")
                    for i in v:
                        model = os.path.join(get_path, i)
                        if not os.path.exists(model):
                            # shutil.copyfile(os.path.join(sour_get_path, i), model)
                            dest_sftp.get(os.path.join(sour_get_path, i), model)
                        di[k] = model
                        models.append(di)
                    mod_li = dest_sftp.listdir(sour_get_path)
                    for a in mod_li:
                        if a.endswith('.json') or a.endswith('.txt'):
                            mp = os.path.join(get_path, a)
                            if not os.path.exists(mp):
                                dest_sftp.get(os.path.join(sour_get_path, a), mp)
                                # shutil.copyfile(os.path.join(sour_get_path, a), mp)
                    if not os.path.exists(os.path.join(map_path, 'label_map.txt')):
                        # shutil.copyfile(os.path.join(sour_map_path, 'label_map.txt'), os.path.join(map_path, 'label_map.txt'))
                        dest_sftp.get(os.path.join(sour_map_path, 'label_map.txt'),
                                        os.path.join(map_path, 'label_map.txt'))
                    dest_scp.close()
                    dest_sftp.close()
                    dest_ssh.close()
                    if task_type == "semantic-segmentation":
                        dest = '/data/deeplearning/train_platform/eva_dest_data/s_data/'
                        if type != u"请选择数据":
                            path = os.path.join(sour, type)
                            if not os.path.exists(path):
                                dest_scp, dest_sftp, dest_ssh = client(id="host")
                                dest_scp.get(path, path, recursive=True)
                                # dest_sftp.get(path, path)
                                dest_scp.close()
                                dest_sftp.close()
                                dest_ssh.close()
                            sour_li.append(os.path.join(sour, type))
                        if l_value:
                            path_list = []
                            sour_dir = '/data/deeplearning/train_platform/eva_sour_data/s_data/test_data'

                            dest_scp, dest_sftp, dest_ssh = client(id="host")
                            test_spath = os.path.join(data0_dest_data, data_name, 'kd_' + data_type + '_test.lst')
                            test_dpath = os.path.join(
                                '/data/deeplearning/train_platform/eva_sour_data/s_data/test_data',
                                'kd_' + data_type + '_test.lst')
                            dest_sftp.get(test_spath, test_dpath)

                            with open(test_dpath, 'r') as f:
                                data = f.readlines()
                            os.remove(test_dpath)

                            for b in data:
                                s = b.split('\t')
                                path_list.append(s[2])
                            for j in path_list:
                                q = j.split('/')
                                d = os.path.join(data0_sour_data, q[-3], q[-2], q[-1])
                                name = q[-1]
                                dest_sftp.get(d, os.path.join(sour_dir, name))
                                if type == u"请选择数据":
                                    dest_sftp.get(d[:-3] + "png", os.path.join(sour_dir, name[:-3] + "png"))
                            dest_scp.close()
                            dest_sftp.close()
                            dest_ssh.close()

                            if sour_dir not in sour_li:
                                sour_li.append(sour_dir)
                        dest_li = os.listdir(dest)
                        if dest_li:
                            for i in dest_li:
                                shutil.rmtree(os.path.join(dest, i))

                    elif task_type == "classification-model":
                        sour_dir = '/data/deeplearning/train_platform/eva_sour_data/c_data/' + k + '/' + type
                        dest = '/data/deeplearning/train_platform/eva_dest_data/c_data/'
                        test_spath = os.path.join(data0_dest_data, data_name, data_name + '_' + type + '.lst')
                        test_dpath = os.path.join('/data/deeplearning/train_platform/eva_sour_data/c_data', k,
                                                  data_name + '_' + type + '.lst')

                        check_dir = '/data/deeplearning/train_platform/eva_sour_data/c_data/' + k
                        if not os.path.exists(check_dir):
                            os.makedirs(check_dir)
                        shutil.copyfile(test_spath, test_dpath)
                        start_unpack_d(test_dpath, check_dir)
                        os.remove(test_dpath)
                        if sour_dir not in sour_li:
                            sour_li.append(sour_dir)
                        dest_li = os.listdir(dest)
                        if dest_li:
                            for i in dest_li:
                                shutil.rmtree(os.path.join(dest, i))
                    elif task_type == "target-detection":
                        dest = '/data/deeplearning/train_platform/eva_dest_data/s_data/'
                        if type != u"请选择数据":
                            sour_li.append(os.path.join(sour, type))
                        if l_value:
                            sour_dir = '/data/deeplearning/train_platform/eva_sour_data/s_data/test_data'
                            test_spath = os.path.join(data0_dest_data, data_name,
                                                      'kd_' + data_type + '_test.lst')
                            test_dpath = os.path.join(
                                '/data/deeplearning/train_platform/eva_sour_data/s_data/test_data',
                                'kd_' + data_type + '_test.lst')
                            path_list = []
                            shutil.copyfile(test_spath, test_dpath)
                            with open(test_dpath, 'r') as f:
                                data = f.readlines()
                            os.remove(test_dpath)
                            for b in data:
                                s = b.split('\t')
                                path_list.append(s[2])
                            for j in path_list:
                                q = j.split('/')
                                d = os.path.join(data0_sour_data, q[-3], q[-2], q[-1])
                                name = q[-1]
                                shutil.copyfile(d, os.path.join(sour_dir, name))
                            if sour_dir not in sour_li:
                                sour_li.append(sour_dir)
                        dest_li = os.listdir(dest)
                        if dest_li:
                            for i in dest_li:
                                shutil.rmtree(os.path.join(dest, i))
                # else:
                #     if type != u"请选择数据":
                #         task_type = "semantic-segmentation"
                #         sour_li.append(os.path.join(sour, type))
                #         dest = '/data/deeplearning/train_platform/eva_dest_data/s_data/'
                #         dest_li = os.listdir(dest)
                #         if dest_li:
                #             for i in dest_li:
                #                 shutil.rmtree(os.path.join(dest, i))
                #         dest_scp, dest_sftp, dest_ssh = client(id='0')
                #         model_dict = {}
                #         name_li = [
                #             'virtual-lane', 'sign-pspnet', 'sign-mask-rcnn', 'pole-mxnet', 'lane-resnet', 'flownet',
                #             'arrow-classfication'
                #         ]
                #         for i in name_li:
                #             li = dest_sftp.listdir(os.path.join('/data12_1/model', i))
                #             model_dict[i] = li
                #         for o, p in model_dict.items():
                #             if k in p:
                #                 r_path = os.path.join('/data12_1/model', o, k)
                #         get_path = '/data/deeplearning/train_platform/train_task/' + k + '/output/models/'
                #         if not os.path.exists(get_path):
                #             os.makedirs(get_path)
                #         for i in v:
                #             model = os.path.join(r_path, i)
                #             if not os.path.exists(os.path.join(get_path, i)):
                #                 dest_sftp.get(model, os.path.join(get_path, i))
                #             di[k] = os.path.join(get_path, i)
                #             models.append(di)
                #         mod_li = dest_sftp.listdir(r_path)
                #         for a in mod_li:
                #             if a.endswith('.json'):
                #                 if not os.path.exists(os.path.join(get_path, a)):
                #                     dest_sftp.get(os.path.join(r_path, a), os.path.join(get_path, a))
        return sour_li, dest, models, task_type
    else:
        return "", "", "", ""


def copy_trt_data(l_value, dicts, mach_id, type):
    models = []
    if dicts:
        if l_value:
            lis = os.listdir('/data/deeplearning/train_platform/trt_eval_sour/s_data')
            if len(lis) > 0:
                for j in lis:
                    os.remove(os.path.join('/data/deeplearning/train_platform/trt_eval_sour/s_data', j))

        sour = '/data/deeplearning/train_platform/trt_eval_sour/s_data'
        sour_li = []
        for k, v in dicts.items():
            if v:
                train_task = db.session.query(TrainTask).filter_by(task_name=k).first()
                di = {}
                if train_task:
                    task_type = train_task.type
                    machine_id = train_task.machine_id
                    data_desc_id = train_task.data_desc_id
                    datas = db.session.query(Datas).filter_by(id=data_desc_id).first()
                    data_name = datas.data_name
                    data_type = datas.data_type
                    get_path = '/data/deeplearning/train_platform/train_task/' + k + '/output/models/'
                    map_path = '/data/deeplearning/train_platform/train_task/' + k + '/conf/'
                    if not os.path.exists(get_path):
                        os.makedirs(get_path)
                    if not os.path.exists(map_path):
                        os.makedirs(map_path)
                    if machine_id != mach_id:
                        dest_scp, dest_sftp, dest_ssh = client(id="host")
                        for i in v:
                            model = os.path.join(get_path, i)
                            if not os.path.exists(model):
                                dest_sftp.get(model, model)
                            di[k] = model
                            models.append(di)
                        mod_li = dest_sftp.listdir(get_path)
                        for a in mod_li:
                            if a.endswith('.json') or a.endswith('.txt'):
                                mp = os.path.join(get_path, a)
                                if not os.path.exists(mp):
                                    dest_sftp.get(mp, mp)
                        dest_sftp.get(os.path.join(map_path, 'label_map.txt'), os.path.join(map_path, 'label_map.txt'))
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()
                    else:
                        for i in v:
                            model = os.path.join(get_path, i)
                            di[k] = model
                            models.append(di)
                    if task_type == "semantic-segmentation":
                        dest = '/data/deeplearning/train_platform/trt_eval_dest/s_data'
                        if type != u"请选择数据":
                            sour_li.append(os.path.join(sour, type))
                        if l_value:
                            sour_dir = '/data/deeplearning/train_platform/trt_eval_sour/s_data'
                            if machine_id != mach_id:
                                dest_scp, dest_sftp, dest_ssh = client(id="host")
                                test_spath = os.path.join('/data/deeplearning/train_platform/data', data_name,
                                                          'kd_' + data_type + '_test.lst')
                                test_dpath = os.path.join('/data/deeplearning/train_platform/trt_eval_sour/s_data',
                                                          'kd_' + data_type + '_test.lst')
                                path_list = []
                                dest_sftp.get(test_spath, test_dpath)
                                with open(test_dpath, 'r') as f:
                                    data = f.readlines()
                                for b in data:
                                    s = b.split('\t')
                                    path_list.append(s[2])
                                for j in path_list:
                                    q = j.split('/')
                                    name = q[-1]
                                    dest_sftp.get(j, os.path.join(sour_dir, name))
                                os.remove(test_dpath)
                                dest_scp.close()
                                dest_sftp.close()
                                dest_ssh.close()
                            else:
                                path_list = []
                                with open(
                                        os.path.join('/data/deeplearning/train_platform/data', data_name,
                                                     'kd_' + data_type + '_test.lst'), 'r') as f:
                                    data = f.readlines()
                                for b in data:
                                    s = b.split('\t')
                                    path_list.append(s[2])
                                    path_list.append(s[3].strip())

                                for j in path_list:
                                    q = j.split('/')
                                    name = q[-1]
                                    open(os.path.join(sour_dir, name), "wb").write(open(j, "rb").read())
                            if sour_dir not in sour_li:
                                sour_li.append(sour_dir)
                        dest_li = os.listdir(dest)
                        if dest_li:
                            for i in dest_li:
                                shutil.rmtree(os.path.join(dest, i))
                    elif task_type == "classification-model":
                        sour_dir = '/data/deeplearning/train_platform/trt_eval_sour/c_data/' + k + '/' + type
                        dest = '/data/deeplearning/train_platform/eva_dest_data/c_data/'
                        if machine_id != mach_id:
                            dest_scp, dest_sftp, dest_ssh = client(id="host")
                            test_spath = os.path.join('/data/deeplearning/train_platform/data', data_name,
                                                      data_name + '_' + type + '.lst')
                            test_dpath = os.path.join('/data/deeplearning/train_platform/trt_eval_sour/c_data', k,
                                                      data_name + '_' + type + '.lst')

                            check_dir = '/data/deeplearning/train_platform/trt_eval_sour/c_data/' + k
                            if not os.path.exists(check_dir):
                                os.makedirs(check_dir)
                            dest_sftp.get(test_spath, test_dpath)
                            start_unpack_d(test_dpath, check_dir, dest_sftp)
                            os.remove(test_dpath)
                            dest_scp.close()
                            dest_sftp.close()
                            dest_ssh.close()
                        else:
                            test_spath = os.path.join('/data/deeplearning/train_platform/data/', data_name,
                                                      data_name + '_' + type + '.lst')
                            check_dir = '/data/deeplearning/train_platform/trt_eval_sour/c_data/' + k
                            if not os.path.exists(check_dir):
                                os.makedirs(check_dir)
                            start_unpack(test_spath, check_dir)
                        if sour_dir not in sour_li:
                            sour_li.append(sour_dir)
                        dest_li = os.listdir(dest)
                        if dest_li:
                            for i in dest_li:
                                shutil.rmtree(os.path.join(dest, i))
        return sour_li, dest, models, task_type
    else:
        return "", "", "", ""


def get_datas(taskid, sour_data, data_name):
    dest_scp, dest_sftp, dest_ssh = client(id="host")
    if taskid:
        _data = BaseDataTask()
        _data.data_status(taskid)
        data_info = _data.process[0]
        if data_info["type"] == "semantic-segmentation":
            sour_datas = data_info["sour_data"].split(",")
            for s_data in sour_datas:
                if s_data:
                    down_path = os.path.join(data0_sour_data,s_data)
                    dest_path = os.path.join(data_sour_data, s_data)
                    if not os.path.exists(dest_path):
                        dest_scp.get(down_path, dest_path, recursive=True)
            down_path = os.path.join(data0_dest_data, data_info["data_name"])
            files = dest_sftp.listdir(down_path)
            for i in files:
                if i.endswith("lst"):
                    dest_sftp.get(os.path.join(down_path, i), os.path.join(data_dest_data, data_name, i))
    down_path = os.path.join(data0_sour_data, sour_data)
    dest_path = os.path.join(data_sour_data, sour_data)
    if os.path.exists(dest_path):
        files_li = os.listdir(down_path)
        files_lis = dest_sftp.listdir(dest_path)
        files_li.sort()
        files_lis.sort()
        if len(files_li) != len(files_lis):
            shutil.rmtree(dest_path)
            dest_scp.get(down_path, dest_path, recursive=True)
    else:
        dest_scp.get(down_path, dest_path, recursive=True)
    dest_scp.close()
    dest_sftp.close()
    dest_ssh.close()