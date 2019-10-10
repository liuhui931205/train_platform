# -*-coding:utf-8-*-
from .base_score import BaseScoreTask
import multiprocessing
import os
from flask import current_app
import json
import time


class ScoreTasks(BaseScoreTask):

    def __init__(self):
        super(BaseScoreTask, self).__init__()
        self.error_code = 1
        self.message = 'Autoscore start'
        self.lock = multiprocessing.Lock()
        self.task_id = ""
        self.status = ""

    def do_task(self, task_id, area_name, scence_name, gpus, weights_dir, img, status, host):
        sour_path = os.path.join("/data/deeplearning/train_platform/recog_rate_sour", area_name, scence_name)
        dest_path = os.path.join("/data/deeplearning/train_platform/recog_rate_dest", area_name, scence_name)
        self.task_id = task_id
        if not os.path.exists(sour_path):
            self.error_code = 0
            self.message = 'Autoeva start failed'
        else:
            task_name = ""
            for model in weights_dir:
                for l, j in model.items():
                    task_name += l + ","
            self.start(task_id, area_name, scence_name,task_name, gpus, img, status, host)
            pro = multiprocessing.Process(target=self.do_asyn, args=(sour_path, dest_path, gpus, weights_dir, img))
            pro.start()
            self.error_code = 1
            self.message = 'Autoeva start success'

    def do_asyn(self, sour_path, dest_path, gpus, model_path, img):
        from Apps.libs.s_model_predict_up.eval_score import Task, do_seg

        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        count = manager.Value("i", 0)

        if not os.path.exists(model_path):
            current_app.logger.warning("model[{}] is not exist".format(model_path))
            exit(0)
        # model_net = ModelResNetRoad(gpu_id=7)
        json_path = os.path.dirname(model_path)
        json_path = os.path.join(json_path, 'map_label.json')
        with open(json_path, 'r') as f:
            datas = f.read()
        seg_label_li = json.loads(datas)
        cls = len(seg_label_li) - 2
        if not os.path.exists(dest_path):
            os.mkdir(dest_path)

        files = os.listdir(sour_path)

        for i in files:
            if i.endswith("jpg"):
                task_ = Task(img_name=os.path.join(sour_path, i), dest_path=os.path.join(dest_path, i[:-3] + "png"))
                task_queue.put(task_)

        gpus = gpus
        gpu_ids = gpus.split(",")
        gpu_ids.sort(reverse=True)
        gpu_count = len(gpu_ids)
        single_count = int(2)
        gpu_threads = []
        total_count = task_queue.qsize()
        pro = multiprocessing.Process(target=self.update_task, args=(count, total_count))
        pro.start()

        for i in range(gpu_count * single_count):
            gpu_id = int(gpu_ids[int(i / single_count)])
            t = multiprocessing.Process(target=do_seg,
                                        args=(img, gpu_id, task_queue, model_path, count, cls, seg_label_li,
                                              self.lock))
            t.daemon = True
            gpu_threads.append(t)

        for j in range(single_count):
            for i in range(gpu_count):
                t = gpu_threads[i * single_count + j]
                t.start()
            time.sleep(15)

        for process in gpu_threads:
            process.join()

    def multi_do_asyn(self, sour_path, dest_path, gpus, weights_dir, img):
        from Apps.libs.s_model_predict_up.eval_score import Task, do_seg
        single_gpu = 2
        manager = multiprocessing.Manager()
        task_queue = manager.Queue()
        count = manager.Value('i', 0)
        img_size = len(os.listdir(sour_path))
        mod_size = len(weights_dir)
        total_count = img_size * mod_size
        pro = multiprocessing.Process(target=self.update_task, args=(count, total_count))
        pro.start()
        for model in weights_dir:
            for l, j in model.items():
                task_name = l
                model_path = j
            if dest_path:
                if not os.path.exists(dest_path):
                    os.makedirs(dest_path)
            if not os.path.exists(model_path):
                current_app.logger.warning("model[{}] is not exist".format(model_path))
                exit(0)
            # model_net = ModelResNetRoad(gpu_id=7)
            json_path = os.path.dirname(model_path)
            json_path = os.path.join(json_path, 'map_label.json')
            with open(json_path, 'r') as f:
                datas = f.read()
            seg_label_li = json.loads(datas)
            # for k, v in seg_label_li.items():
            #     if int(k) != 255 and int(k) != 254:
            #         label_li.append(k)
            cls = len(seg_label_li) - 2
            dest_dir = os.path.join(dest_path, task_name)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            # if os.path.exists(s_desc_file):
            #     shutil.copy(s_desc_file, d_desc_file)
            files = os.listdir(sour_path)

            for i in files:
                if i.endswith("jpg"):
                    task_ = Task(img_name=os.path.join(sour_path, i), dest_path=os.path.join(dest_dir, i[:-3] + "png"))
                    task_queue.put(task_)

            gpus = gpus
            gpu_ids = gpus.split(",")
            gpu_ids.sort(reverse=True)
            gpu_count = len(gpu_ids)
            single_count = int(single_gpu)
            gpu_threads = []
            for i in range(gpu_count * single_count):
                gpu_id = int(gpu_ids[int(i / single_count)])
                t = multiprocessing.Process(target=do_seg,
                                            args=(img, gpu_id, task_queue, model_path, count, cls, seg_label_li,
                                                  self.lock))
                t.daemon = True
                gpu_threads.append(t)

            for j in range(single_count):
                for i in range(gpu_count):
                    t = gpu_threads[i * single_count + j]
                    t.start()
                time.sleep(15)

            for process in gpu_threads:
                process.join()
        pro.join()

    def update_task(self, count, total_count):
        while True:
            plan = (count.value / (total_count * 1.00)) * 100
            if plan != 100:
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'unfinshed'
                value = self.callback(self.task_id, self.status)
            if value:
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(6)
