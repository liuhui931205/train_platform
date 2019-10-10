# -*-coding:utf-8-*-
import mxnet as mx
import os
import multiprocessing
from flask import current_app
from Apps.libs.make_data import c_make_record_file, s_make_record_file, t_make_record_file, mask_rcnn_rpn_im2rec
from .base_data import BaseDataTask
from Apps.libs.statistical_data_class import do_data_count
import time
import shutil
from config import data0_dest_data,data_sour_data,data_dest_data
from Apps.utils.copy_all import copyFiles
from Apps.utils.client import client


class DataTasks(BaseDataTask):

    def __init__(self, manager):
        super(DataTasks, self).__init__()
        self.error_code = 1
        self.message = 'Autosele start'
        self.task_id = ''
        self.status = ''
        self._manager = manager
        self.lock = multiprocessing.Lock()

        self._total_count = self._manager.Value("i", 0)
        self._cur_count = self._manager.Value("i", 0)

    def create_data(self, taskid, task_id, data_name, data_type, train, val, test, sour_data, data_describe, status,
                    thread, l_value, types, image_type):

        self.task_id = task_id
        data_path = os.path.join(data_dest_data, data_name)
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        pro = multiprocessing.Process(target=self.create_async,
                                      args=(taskid, data_name, data_path, data_type, sour_data, train, test, thread,
                                            l_value, types, image_type))

        self.start(taskid, task_id, data_name, data_type, train, val, test, sour_data, data_describe, status, types,
                   image_type)
        pro.start()
        self.error_code = 1
        self.message = 'data create start '

    def create_async(self, taskid, data_name, data_path, data_type, sour_data, train, test, thread, l_value, types,
                     image_type):
        if types == "semantic-segmentation":
            data_path = data_path + '/kd_' + data_type
            sour_data = os.path.join(data_sour_data, sour_data)
            train_ratio = float(train)
            test_ratio = float(test)
            thread = int(thread)
            chunks = 1
            resize = 0
            quality = 95
            shuffle = True
            pass_through = False
            center_crop = False
            center_pad = False
            pack_label = False
            color = -1
            exts = ['.jpeg', '.jpg', '.png']
            encoding = '.jpg'
            recursive = False

            if not os.path.isdir(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            if l_value or taskid:
                s_make_record_file.make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks,
                                                 taskid)
            path = os.path.dirname(data_path)
            do_data_count(path, image_type)
            if os.path.isdir(data_path):
                working_dir = data_path
            else:
                working_dir = os.path.dirname(data_path)
            files = [
                os.path.join(working_dir, fname)
                for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))
            ]
            count = 0
            coun = 0
            for fname in files:
                couns = os.popen('cat ' + fname + '| wc -l').readlines()[0]
                coun += int(couns.strip())
            self._total_count.value = coun
            cou_pro = multiprocessing.Process(target=self.update_task, args=(sour_data, data_name))
            cou_pro.start()
            for fname in files:
                if fname.startswith(data_path) and fname.endswith('.lst'):
                    # current_app.logger.info('Creating .rec file from', fname, 'in', working_dir)
                    print('Creating .rec file from', fname, 'in', working_dir)
                    count += 1
                    image_list = s_make_record_file.read_list(fname)
                    # -- write_record -- #
                    if thread > 1 and multiprocessing is not None:
                        q_in = [multiprocessing.Queue(1024) for i in range(thread)]
                        q_out = multiprocessing.Queue(1024)
                        read_process = [
                            multiprocessing.Process(target=s_make_record_file.read_worker,
                                                    args=(pass_through, center_crop, center_pad, resize, quality,
                                                          q_in[i], q_out)) for i in range(thread)
                        ]
                        for p in read_process:
                            p.start()
                            # p.daemon = True
                        write_process = multiprocessing.Process(target=s_make_record_file.write_worker,
                                                                args=(q_out, fname, working_dir, self.lock,
                                                                      self._cur_count))

                        write_process.start()
                        # write_process.daemon = True
                        for i, item in enumerate(image_list):
                            q_in[i % len(q_in)].put((i, item))
                        for q in q_in:
                            q.put(None)
                        for p in read_process:
                            p.join()
                        q_out.put(None)
                        write_process.join()

                    else:
                        current_app.logger.info('multiprocessing not available, fall back to single threaded encoding')
                        try:
                            import Queue as queue
                        except ImportError:
                            import queue
                        q_out = queue.Queue()
                        fname = os.path.basename(fname)
                        fname_rec = os.path.splitext(fname)[0] + '.rec'
                        fname_idx = os.path.splitext(fname)[0] + '.idx'

                        record = mx.seg_recordio.MXIndexedSegRecordIO(os.path.join(working_dir, fname_idx),
                                                                      os.path.join(working_dir, fname_rec), 'w')
                        cnt = 0
                        pre_time = time.time()
                        for i, item in enumerate(image_list):
                            s_make_record_file.image_encode(pass_through, center_crop, center_pad, resize, quality, i,
                                                            item, q_out)
                            if q_out.empty():
                                continue
                            _, s, _ = q_out.get()
                            record.write_idx(item[0], s)
                            if cnt % 100 == 0:
                                cur_time = time.time()
                                current_app.logger.info('time:', cur_time - pre_time, ' count:', cnt)
                                pre_time = cur_time
                            cnt += 1
            cou_pro.join()
            if not count:
                current_app.logger.warning('Did not find and list file with prefix %s' % data_path)
        elif types == "classification-model":
            data_path = data_path + "/" + data_name
            sour_data = os.path.join(data_sour_data, sour_data)
            train_ratio = float(train)
            test_ratio = float(test)
            thread = int(thread)
            chunks = 1
            resize = 256
            quality = 95
            shuffle = True
            pass_through = False
            center_crop = False
            center_pad = True
            pack_label = False
            color = -1
            exts = ['.jpeg', '.jpg', '.png']
            encoding = '.jpg'
            recursive = False

            if not os.path.isdir(os.path.dirname(data_path)):
                os.makedirs(os.path.dirname(data_path))
            if l_value:
                c_make_record_file.make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks)
            if os.path.isdir(data_path):
                working_dir = data_path
            else:
                working_dir = os.path.dirname(data_path)
            files = [
                os.path.join(working_dir, fname)
                for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))
            ]
            count = 0
            coun = 0
            for fname in files:
                couns = os.popen('cat ' + fname + '| wc -l').readlines()[0]
                coun += int(couns.strip())
            self._total_count.value = coun
            cou_pro = multiprocessing.Process(target=self.update_task)
            cou_pro.start()
            for fname in files:
                if fname.startswith(data_path) and fname.endswith('.lst'):
                    print('Creating .rec file from', fname, 'in', working_dir)
                    count += 1
                    image_list = c_make_record_file.read_list(fname)
                    # -- write_record -- #
                    if thread > 1 and multiprocessing is not None:
                        q_in = [multiprocessing.Queue(1024) for i in range(thread)]
                        q_out = multiprocessing.Queue(1024)
                        read_process = [
                            multiprocessing.Process(target=c_make_record_file.read_worker,
                                                    args=(q_in[i], sour_data, pass_through, center_crop, center_pad,
                                                          resize, quality, q_out, pack_label, color, encoding))
                            for i in range(thread)
                        ]
                        for p in read_process:
                            p.start()
                            # p.daemon = True
                        write_process = multiprocessing.Process(target=c_make_record_file.write_worker,
                                                                args=(q_out, fname, working_dir, self.lock,
                                                                      self._cur_count))

                        write_process.start()
                        # write_process.daemon = True
                        for i, item in enumerate(image_list):
                            q_in[i % len(q_in)].put((i, item))
                        for q in q_in:
                            q.put(None)
                        for p in read_process:
                            p.join()
                        q_out.put(None)
                        write_process.join()

                    else:
                        print('multiprocessing not available, fall back to single threaded encoding')
                        try:
                            import Queue as queue
                        except ImportError:
                            import queue
                        q_out = queue.Queue()
                        fname = os.path.basename(fname)
                        fname_rec = os.path.splitext(fname)[0] + '.rec'
                        fname_idx = os.path.splitext(fname)[0] + '.idx'
                        record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                               os.path.join(working_dir, fname_rec), 'w')
                        cnt = 0
                        pre_time = time.time()
                        for i, item in enumerate(image_list):
                            c_make_record_file.image_encode(pass_through, center_crop, center_pad, resize, quality, i,
                                                            item, q_out)
                            if q_out.empty():
                                continue
                            _, s, _ = q_out.get()
                            record.write_idx(item[0], s)
                            if cnt % 1000 == 0:
                                cur_time = time.time()
                                current_app.logger.info('time:', cur_time - pre_time, ' count:', cnt)
                                pre_time = cur_time
                            cnt += 1
            cou_pro.join()
            if not count:
                print('Did not find and list file with prefix %s' % data_path)
        elif types == "target-detection":
            # data_path = data_path + '/kd_' + data_type
            sour_data = os.path.join(data_sour_data, sour_data)
            train_ratio = float(train)
            test_ratio = float(test)
            thread = int(thread)
            chunks = 1
            resize = 0
            quality = 95
            shuffle = True
            pass_through = False
            center_crop = False
            center_pad = False
            pack_label = False
            color = 1
            exts = ['.jpeg', '.jpg', '.png']
            encoding = '.jpg'
            recursive = False

            # if os.path.exists(os.path.join(data_path,"data")):
            #     os.makedirs(os.path.join(data_path,"data"))
            shutil.copytree(sour_data, os.path.join(data_path, "data"), True)
            manager = multiprocessing.Manager()
            lock = manager.Lock()
            task_queue = manager.Queue()
            count_value = manager.Value("i", 0)

            datas_path = os.path.join(data_path, "data")
            dirs = os.listdir(datas_path)
            for d in dirs:
                sub_dir = os.path.join(datas_path, d)
                if not os.path.isdir(sub_dir):
                    continue
                # json_file = os.path.join(sub_dir, "annot_" + d + ".json")
                json_file = os.path.join(sub_dir, d + ".json")
                if not os.path.exists(json_file):
                    continue
                task = t_make_record_file.Task(json_file)
                task_queue.put(task)
            for i in range(10):
                task = t_make_record_file.Task(None, exit_flag=True)
                task_queue.put(task)
            all_pros = []
            for i in range(10):
                do_pro = multiprocessing.Process(target=t_make_record_file.do, args=(task_queue,))
                all_pros.append(do_pro)
            for i in all_pros:
                i.start()
            for i in all_pros:
                i.join()

            sour_data = os.path.join(data_path, "data")
            data_path = data_path + '/masks_datas'
            if l_value:
                # make_list(args)
                mask_rcnn_rpn_im2rec.make_list_dir(data_path, sour_data, shuffle, train_ratio, test_ratio, chunks)
            if os.path.isdir(data_path):
                working_dir = data_path
            else:
                working_dir = os.path.dirname(data_path)
            files = [
                os.path.join(working_dir, fname)
                for fname in os.listdir(working_dir)
                if os.path.isfile(os.path.join(working_dir, fname))
            ]
            count = 0
            coun = 0
            for fname in files:
                couns = os.popen('cat ' + fname + '| wc -l').readlines()[0]
                coun += int(couns.strip())
            self._total_count.value = coun
            cou_pro = multiprocessing.Process(target=self.update_task, args=(sour_data, data_name, types, count_value))
            cou_pro.start()
            for fname in files:
                if fname.startswith(data_path) and fname.endswith('.lst'):
                    current_app.logger.info('Creating .rec file from', fname, 'in', working_dir)
                    count += 1
                    image_list = mask_rcnn_rpn_im2rec.read_list(fname)
                    # -- write_record -- #
                    if thread > 1 and multiprocessing is not None:
                        q_in = [multiprocessing.Queue(1024) for i in range(thread)]
                        q_out = multiprocessing.Queue(1024)
                        read_process = [multiprocessing.Process(target=mask_rcnn_rpn_im2rec.read_worker, args=(
                            count_value, lock, pass_through, center_crop, encoding, resize, quality, color, q_in[i], q_out)) \
                                        for i in range(thread)]
                        for p in read_process:
                            p.start()
                        write_process = multiprocessing.Process(target=mask_rcnn_rpn_im2rec.write_worker,
                                                                args=(q_out, fname, working_dir, self.lock,
                                                                      self._cur_count))
                        write_process.start()

                        for i, item in enumerate(image_list):
                            q_in[i % len(q_in)].put((i, item))
                        for q in q_in:
                            q.put(None)
                        for p in read_process:
                            p.join()

                        q_out.put(None)
                        write_process.join()
                    else:
                        current_app.logger.info('multiprocessing not available, fall back to single threaded encoding')
                        try:
                            import Queue as queue
                        except ImportError:
                            import queue
                        q_out = queue.Queue()
                        fname = os.path.basename(fname)
                        fname_rec = os.path.splitext(fname)[0] + '.rec'
                        fname_idx = os.path.splitext(fname)[0] + '.idx'
                        record = mx.recordio.MXIndexedRecordIO(os.path.join(working_dir, fname_idx),
                                                               os.path.join(working_dir, fname_rec), 'w')
                        cnt = 0
                        pre_time = time.time()
                        for i, item in enumerate(image_list):
                            mask_rcnn_rpn_im2rec.image_encode(pass_through, center_crop, encoding, resize, quality,
                                                              color, i, item, q_out)
                            if q_out.empty():
                                continue
                            _, s, _ = q_out.get()
                            record.write_idx(item[0], s)
                            if cnt % 1000 == 0:
                                cur_time = time.time()
                                current_app.logger.info('time:', cur_time - pre_time, ' count:', cnt)
                                pre_time = cur_time
                            cnt += 1
                if not count:
                    current_app.logger.warning('Did not find and list file with prefix %s' % data_path)

            cou_pro.join()

    def update_task(self, sour_data=None, data_name=None, types=None, count=None):
        while True:
            print types
            if types == "target-detection":
                print "error:{},cur:{}".format(count.value, self._cur_count.value)
                if int((count.value + self._cur_count.value) / float(self._total_count.value)) != 1:
                    self.status = (count.value + self._cur_count.value) * 100 / float(self._total_count.value)
                    value = self.callback(self.task_id, self.status)
                else:
                    self.status = 'completed!'
                    value = self.callback(self.task_id, self.status)
                if value:
                    current_app.logger.info('---------------finished---------------')
                    break
                time.sleep(6)
            else:
                if self._cur_count.value != 0:
                    plan = (self._cur_count.value / (self._total_count.value * 1.00)) * 100
                    if plan != 100:
                        self.status = plan
                        value = self.callback(self.task_id, self.status)
                    else:
                        self.status = 'completed!'
                        value = self.callback(self.task_id, self.status)

                    if value:
                        current_app.logger.info('---------------finished---------------')
                        shutil.rmtree(sour_data)

                        dest_scp, dest_sftp, dest_ssh = client(id="host")
                        dest_path = os.path.join(data0_dest_data, data_name)
                        dirs = dest_sftp.listdir(data0_dest_data)
                        paths = os.path.join(data_dest_data, data_name)
                        if data_name not in dirs:
                            dest_sftp.mkdir(dest_path)
                        files = os.listdir(paths)
                        for i in files:
                            dest_sftp.put(os.path.join(paths, i), os.path.join(dest_path, i))
                        dest_scp.close()
                        dest_sftp.close()
                        dest_ssh.close()

                        break
                time.sleep(6)
