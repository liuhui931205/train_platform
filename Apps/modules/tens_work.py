# -*-coding:utf-8-*-
import os
import multiprocessing
from .base_eva import BaseEvaTask
import time
import json
from flask import current_app
import shutil
from Apps.libs.c_model_predict.val_class_2 import ModelClassArrow
from Apps.libs.tensor.eval_trt import EvalData
import cv2


class TenTasks(BaseEvaTask):

    def __init__(self):
        super(TenTasks, self).__init__()
        self.error_code = 1
        self.message = 'Autoeva start'
        self.task_id = ''
        self.status = ''
        self.total_count = 0
        self.lock = multiprocessing.Lock()

    def evaluating(self, task_id, sour_dir, gpus, dest_dir, single_gpu, model, status, task_type, img, argmax,
                   effective_c):
        self.task_id = task_id
        self.start(task_id, sour_dir, gpus, dest_dir, single_gpu, model, status)
        if task_type == "semantic-segmentation":
            pro = multiprocessing.Process(
                target=self.s_evaluat_async,
                args=(sour_dir, gpus, dest_dir, model, img, effective_c))
            pro.start()
        elif task_type == "classification-model":
            pro = multiprocessing.Process(target=self.c_evaluat_async, args=(sour_dir, gpus, dest_dir, model))
            pro.start()
        self.error_code = 1
        self.message = 'Autoeva start'

    def s_evaluat_async(self, sour_dir, gpus, dest_dirs, models, img, effective_c):
        if img == "half":
            input_size = (1, 3, 512, 1224)
            output_size = (512, 1224)
            height = 1024
            width = 1224
        else:
            input_size = (1, 3, 1024, 1224)
            output_size = (1024, 1224)
            height = 1024
            width = 1224
        manager = multiprocessing.Manager()
        count = manager.Value('i', 0)
        img_size = 0
        for i in sour_dir:
            li = os.listdir(i)
            img_size += len(li)/2
        mod_size = len(models)
        self.total_count = img_size * mod_size
        pro = multiprocessing.Process(target=self.update_task, args=(count,))
        pro.start()
        for model in models:
            task_name = ""
            for l, j in model.items():
                task_name = l
                s = j.split('.')
                index = s[0][-4:]
                index = l + '_' + str(index)
                model_path = j

            if dest_dirs:
                if not os.path.exists(dest_dirs):
                    os.makedirs(dest_dirs)
            if not os.path.exists(model_path):
                current_app.logger.warning("model[{}] is not exist".format(model_path))
                exit(0)
            # model_net = ModelResNetRoad(gpu_id=7)
            json_path = os.path.dirname(model_path)
            json_path = os.path.join(json_path, 'map_label.json')
            map_path = "/data/deeplearning/train_platform/train_task/" + task_name + "/conf/label_map.txt"
            label_map_data = []
            with open(json_path, 'r') as f:
                datas = f.read()
            seg_label_li = json.loads(datas)
            with open(map_path, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        line = (line.strip()).split('\t')
                        label_map_data.append(line)
                    else:
                        label_map_data.pop(0)
                        break
            dest_dir = os.path.join(dest_dirs, str(index))
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)
            s_desc_file = "/data/deeplearning/train_platform/train_task/" + task_name + "/output/models/info.txt"
            d_desc_file = os.path.join(dest_dir, "info.txt")
            if os.path.exists(s_desc_file):
                shutil.copy(s_desc_file, d_desc_file)
            gpus = gpus
            gpu_ids = gpus.split(",")
            gpu_ids.sort(reverse=True)
            eval_data = EvalData(
                sour_dir,
                dest_dir,
                model_path,
                input_size,
                output_size,
                count,
                effective_c=int(effective_c),
                gpu_id=int(gpu_ids[0]),
                batch_size=1,
                height=height,
                width=width,
                use_half_image=False,
                save_result_or_not=True,
                iou_thresh_low=0.5,
                min_pixel_num=2000,
                flip=False,
                upsample_or_not=False,
                seg_label_li=seg_label_li,
                label_map_data=label_map_data)
            eval_data.run()

    def c_evaluat_async(self, sour_dir, gpus, dest_dirs, models):
        manager = multiprocessing.Manager()
        count = manager.Value('i', 0)
        img_size = os.popen("ls -lR " + sour_dir[0] + "|grep jpg| wc -l")
        mod_size = len(models)
        self.total_count = int(img_size.readline().strip()) * mod_size
        pro = multiprocessing.Process(target=self.update_task, args=(count,))
        pro.start()

        image_dir = sour_dir[0]
        for model in models:
            map_path = ""
            for l, j in model.items():
                s = j.split('/')
                index = s[-1][:-7]
                model_file = j
                map_path = "/data/deeplearning/train_platform/train_task/" + l + "/conf/label_map.txt"
            if dest_dirs:
                if not os.path.exists(os.path.join(dest_dirs, index)):
                    os.makedirs(os.path.join(dest_dirs, index))
            if not os.path.exists(model_file):
                current_app.logger.error("model[{}] is not exist".format(model_file))

            name_dict = {}
            label_dict = {}
            # for label in arrow_labels_v4_test:
            #     if label.categoryId not in name_dict:
            #         name_dict[label.categoryId] = label.name
            #         name_dict[label.label] = label.name
            #     if label.categoryId not in label_dict:
            #         label_dict[label.categoryId] = label.label
            li = []
            with open(map_path, 'r') as f:
                while True:
                    line = f.readline()
                    if line:
                        line = line.split('\t')
                        li.append(line)
                    else:
                        break
            for i in li:
                if li.index(i) != 0:
                    if i[0] != '\n':
                        if int(i[1]) not in name_dict:
                            name_dict[int(i[1])] = str(i[3])
                            name_dict[str(i[2])] = str(i[3])
                        if int(i[1]) not in label_dict:
                            label_dict[int(i[1])] = str(i[2])

            if not os.path.exists(model_file):
                current_app.logger.error("model file[{}] is not exist".format(model_file))
                exit(0)

            dest_dir = os.path.join(dest_dirs, index, "test_pre")
            gpu_id = int(gpus.split(',')[0])
            model_net = ModelClassArrow(model_file, gpu_id=gpu_id)
            proc_list = []

            current_app.logger.info("loading test label...\n")
            label_map = {}
            total_val = 0
            recall_map = {}

            dir_list = os.listdir(image_dir)
            for id_dir in dir_list:
                if id_dir == '12':
                    print("ok")
                class_dir = os.path.join(image_dir, id_dir)
                file_list = os.listdir(class_dir)
                for file_id in file_list:
                    if not file_id.endswith("jpg"):
                        continue

                    proc_list.append(os.path.join(image_dir, id_dir, file_id))
                    class_id = id_dir
                    label_map[file_id] = class_id
                    total_val += 1

                    if class_id not in recall_map:
                        recall_map[class_id] = {"total": 1}
                    else:
                        recall_map[class_id]["total"] += 1

            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            for id_ in proc_list:
                file_path = id_
                im = cv2.imread(file_path)
                current_app.logger.info("im.shape  {}".format(im.shape))

                try:
                    pred_label = None
                    start = time.time()
                    assert os.path.exists(file_path)
                    with open(file_path, "rb") as f:
                        img = f.read()
                        pred_label, accuracy = model_net.do(count, self.lock, image_data=img)
                    end = time.time()

                    class_id = pred_label[0]
                    if class_id == 12:
                        current_app.logger.info("ok")
                    class_id = label_dict[int(class_id)]
                    class_acc = accuracy[0]
                    class_dir = os.path.join(dest_dir, str(class_id))
                    if not os.path.exists(class_dir):
                        os.makedirs(class_dir)

                    dest_path = os.path.join(class_dir, os.path.basename(id_))
                    shutil.copy(file_path, dest_path)
                    current_app.logger.info("Processed {} in {} ms,acc:{}, labels:{} vs. {}".format(
                        os.path.basename(dest_path), str((end - start) * 1000), class_acc, str(class_id),
                        label_map[os.path.basename(file_path)]))

                except Exception as e:
                    current_app.logger.error(repr(e))

            current_app.logger.info("start to calculate recall and accuracy...")
            current_app.logger.info("loading prediction...")
            accuracy_map = {}

            pred_map = {}
            class_dir_list = os.listdir(dest_dir)
            for class_dir in class_dir_list:
                pred_dir = os.path.join(dest_dir, class_dir)
                pred_list = os.listdir(pred_dir)

                for pred_file in pred_list:
                    if not pred_file.endswith("jpg"):
                        continue
                    pred_map[pred_file] = class_dir

                    if class_dir not in accuracy_map:
                        accuracy_map[class_dir] = {"total": 1}
                    else:
                        accuracy_map[class_dir]["total"] += 1

            correct_map = {}

            for image_name, class_id in label_map.items():
                if class_id == 12:
                    current_app.logger.info("ok")
                pred_class = pred_map[image_name]

                if class_id == pred_class:
                    if class_id not in correct_map:
                        correct_map[class_id] = 1
                    else:
                        correct_map[class_id] += 1

            current_app.logger.info("start to calculate recall and accuracy...")

            for class_id, count in correct_map.items():
                recall_map[class_id]["correct"] = count
                recall_map[class_id]["rate"] = float(count) / float(recall_map[class_id]["total"]) * 100

                accuracy_map[class_id]["correct"] = count
                accuracy_map[class_id]["rate"] = float(count) / float(accuracy_map[class_id]["total"]) * 100

            # format
            info_data = "recall, id, rate, correct/total\n"

            for class_id, info in recall_map.items():
                rate = 0
                if 'rate' in info:
                    rate = info['rate']
                correct = 0
                if 'correct' in info:
                    correct = info['correct']
                total = 0
                if 'total' in info:
                    total = info['total']

                label_name = name_dict[class_id]
                data1 = "{},{},{},{}/{}\n".format(label_name.encode("UTF-8"), class_id, rate, correct, total)
                info_data += data1

            info_data += "\naccuracy, id, rate, correct/total\n"
            for class_id, info in accuracy_map.items():
                rate = 0
                if 'rate' in info:
                    rate = info['rate']
                correct = 0
                if 'correct' in info:
                    correct = info['correct']
                total = 0
                if 'total' in info:
                    total = info['total']

                label_name = name_dict[class_id]
                data2 = "{},{},{},{}/{}\n".format(label_name.encode("UTF-8"), class_id, rate, correct, total)
                info_data += data2

            with open(os.path.join(dest_dirs, index, "info.txt"), 'w') as f:
                f.write(info_data)
        pro.join()

    def update_task(self, count):
        while True:
            plan = (count.value / (self.total_count * 1.00)) * 100
            if plan != 100:
                self.status = plan
                value = self.callback(self.task_id, self.status)
            else:
                self.status = 'completed!'
                time.sleep(10)
                value = self.callback(self.task_id, self.status)
            if value:
                current_app.logger.info('---------------finished---------------')
                break
            time.sleep(6)
